import sys
import os

sys.path.append("../")

import torch
import logging
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import f1_score, average_precision_score
from torch.utils.tensorboard import SummaryWriter
from transformers import GPT2Config, get_linear_schedule_with_warmup
from fastNLP import cache_results, Trainer, Event, TorchDataLoader, EarlyStopCallback, LRSchedCallback

from GPT2_prefix_tuning_with_multi_label_v2.pipe import Pipe, set_seed
from GPT2_prefix_tuning_with_multi_label_v2.prefix_model import Model
from GPT2_prefix_tuning_with_multi_label_v2.utils import load_parameters, pack_labels_batch
from GPT2_prefix_tuning_with_multi_label_v2.metric import RougeMetric, Accuracy

if __name__ == '__main__':
    print('Data and Parameters Initialization...')
    args = load_parameters()
    set_seed(args.seed)

    datafile = "datafile/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-pre_seq_len-{}-lmd-{}-m-loss-{}".format(
        args.dataset, args.known_cls_ratio,
        args.seed, args.lr, args.pre_seq_len, args.lamda_loss, args.m_loss)

    model_filepath = "model/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-pre_seq_len-{}-lmd-{}-m-loss-{}".format(
        args.dataset, args.known_cls_ratio,
        args.seed, args.lr, args.pre_seq_len, args.lamda_loss, args.m_loss)

    if not os.path.exists(datafile):
        os.makedirs(datafile)


    @cache_results("{}/data.pkl".format(datafile), _hash_param=False)
    def load_data(arg):
        pipe = Pipe(arg)
        return pipe


    log_dir = "log/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-pre_seq_len-{}-prefix_projection-{}" \
              "-prefix_hidden_size-{}-prefix_dropout-{}-train_batch_size-{}".format(
        args.dataset, args.known_cls_ratio,
        args.seed, args.lr, args.pre_seq_len, args.prefix_projection, args.prefix_hidden_size, args.prefix_dropout,
        args.train_batch_size)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    writer = SummaryWriter(log_dir=log_dir)
    data = load_data(args)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("train dataset length: {}\t dev dataset length: {}\t test dataset length: {}".format(
        len(data.train_dataset), len(data.eval_dataset), len(data.test_dataset)))

    train_dl = TorchDataLoader(data.train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_dl = TorchDataLoader(data.eval_dataset, batch_size=args.eval_batch_size, shuffle=True)
    # test_dv = TorchDataLoader(data.test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    print('Pre-training begining!')
    # train ind pretrained parameter

    args.num_labels = len(data.known_multi_label_unique)
    # args.len_tokenizer = len(data.tokenizer)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(args)
    model.to(device)

    optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=50,
                                                num_training_steps=len(train_dl) * args.num_train_epochs)

    rougeMetric = RougeMetric(tokenizer=data.tokenizer)
    # accDectection = Accuracy()
    # accDectection_tr = Accuracy()

    best_eval_score = 0
    wait = 0

    for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_steps = 0
        gen_loss, cls_loss, ce_loss = [], [], []
        for step, batch in enumerate(tqdm(train_dl, desc="Train_Iteration")):
            batch.pop("label_text")
            label_ids = batch.pop("label_ids").to(device)
            multi_labels = batch.pop('multi_labels').to(device)
            batch = tuple(pack_labels_batch(t, device) for _, t in batch.items())
            input_ids, input_mask, x_attention_mask, labels = batch

            with torch.set_grad_enabled(True):
                result = model.forward(input_ids=input_ids, attention_mask=input_mask, multi_labels=multi_labels,
                                       labels=labels, x_attention_mask=x_attention_mask, label_ids=label_ids)

                # accDectection_tr.update(logits=result['logits'], label_ids=label_ids)
                loss = result['loss']
                # 记录损失
                gen_loss.append(result['gen_loss'].item())
                cls_loss.append(result['cls_loss'].item())
                ce_loss.append(result['ce_loss'].item())

                optim.zero_grad()
                loss.backward()
                optim.step()
                scheduler.step()

                tr_loss += loss.item()
                nb_tr_steps += 1

        # tr_acc = accDectection_tr.get_metric()
        # logger.info("train_acc: {}".format(tr_acc['acc']))
        # 写进文件
        writer.add_scalar("gen_loss", sum(gen_loss) / len(gen_loss), epoch)
        writer.add_scalar("cls_loss", sum(cls_loss) / len(gen_loss), epoch)

        y_pred = []
        y_label = []
        model.eval()
        for step, batch in enumerate(tqdm(dev_dl, desc="Dev_Iteration")):
            label_text = batch.pop("label_text")
            label_ids = batch.pop("label_ids").to(device)
            multi_labels = batch.pop('multi_labels').to(device)
            batch = tuple(pack_labels_batch(t, device)[0] for _, t in batch.items())
            input_ids, input_mask, x_attention_mask, labels = batch
            with torch.set_grad_enabled(False):
                result = model.evaluate_step(input_ids=input_ids, attention_mask=input_mask,
                                             x_attention_mask=x_attention_mask)
                # 计算 metric
                rougeMetric.update(label_text=label_text, decoder_tokens=result['decoder_tokens'])
                # accDectection.update(logits=result['logits'], label_ids=label_ids)
                y_pred.append(torch.nn.Sigmoid()(result['logits']))
                y_label.append(multi_labels)

        rouge_result = rougeMetric.get_metric()
        # acc_result = accDectection.get_metric()
        y_label = torch.cat(y_label, dim=0).cpu().numpy()
        y_pred = torch.cat(y_pred, dim=0).cpu().numpy()

        # eval_score = rouge_result['rouge_score']
        # eval_score = acc_result['acc']
        pred = np.rint(y_pred)
        eval_score = f1_score(y_label, pred, average='samples')

        if eval_score > best_eval_score:
            wait = 0
            best_eval_score = eval_score
            # 每次保存最好的模型， 重复覆盖
            if not os.path.exists(model_filepath):
                os.makedirs(model_filepath)
            model_name = "checkpoint.pkl"
            model_file = os.path.join(model_filepath, model_name)
            torch.save(model.state_dict(), model_file)
        else:
            wait += 1
            if wait >= args.wait_patient:
                break
        tr_loss /= nb_tr_steps
        logger.info("train_loss: {}".format(tr_loss))
        logger.info("rouge_score: {}".format(rouge_result['rouge_score']))
        logger.info("acc_score: {}".format(eval_score))
        logger.info("gen_loss: {}".format(sum(gen_loss) / nb_tr_steps))
        logger.info("cls_loss: {}".format(sum(cls_loss) / nb_tr_steps))
        logger.info("ce_loss: {}".format(sum(ce_loss) / nb_tr_steps))

    writer.close()
    logger.info('Pre-training finished!')