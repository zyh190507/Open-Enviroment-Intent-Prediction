import sys
import os
import functools

sys.path.append('../')

from tqdm import trange, tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from fastNLP import cache_results, TorchDataLoader, DataSet, Instance
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

from GPT2_full_finetune_with_transformer_layer.model import Model
from GPT2_full_finetune_with_transformer_layer.pipe import Data, set_seed
from GPT2_full_finetune_with_transformer_layer.metric import BoundaryLoss, F_measure, euclidean_metric
from GPT2_full_finetune_with_transformer_layer.utils import load_parameters


class ModelManager:

    def __init__(self, args, data, pretrained_model=None):

        self.model = pretrained_model

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model is None:
            self.model = Model(args)
            self.restore_model(args, self.device)
        self.model.to(self.device)

        self.best_eval_score = 0
        self.delta = None
        self.delta_points = []
        self.centroids = None

        self.test_results = None
        self.predictions = None
        self.true_labels = None

    def open_classify(self, features, data):

        logits = euclidean_metric(features, self.centroids)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
        euc_dis = torch.norm(features - self.centroids[preds], 2, 1).view(-1)
        preds[euc_dis >= self.delta[preds]] = data.unseen_token_id

        return preds

    def evaluation(self, args, data, mode="eval"):
        self.model.eval()

        total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        total_preds = torch.empty(0, dtype=torch.long).to(self.device)
        unkDataset = DataSet()
        if mode == 'eval':
            dataloader = data.eval_dataloader
        elif mode == 'test':
            dataloader = data.test_dataloader
        else:
            return ValueError()

        for batch in tqdm(dataloader, desc="Iteration"):
            batch.pop("label_text")
            batch = tuple(t.to(self.device) for _, t in batch.items())
            input_ids, input_mask, x_attention_mask, labels, label_ids = batch
            with torch.set_grad_enabled(False):
                pooled_output = self.model.evaluate_step(input_ids=input_ids, attention_mask=input_mask,
                                                         x_attention_mask=x_attention_mask)['feature']
                preds = self.open_classify(pooled_output, data)
                if mode == 'test':
                    for input_id, attention_mask, per_label, pred_id, x_att_mask, label_id in zip(
                            input_ids.cpu().numpy(),
                            input_mask.cpu().numpy(),
                            labels.cpu().numpy(),
                            preds.cpu().numpy(),
                            x_attention_mask.cpu().numpy(),
                            label_ids.cpu().numpy()):
                        if pred_id == data.unseen_token_id:
                            # UNK sample
                            unkDataset.append(Instance(input_ids=input_id, attention_mask=attention_mask,
                                                       label_ids=label_id, labels=per_label, x_attention_mask=x_att_mask))
                total_labels = torch.cat((total_labels, label_ids))
                total_preds = torch.cat((total_preds, preds))

        y_pred = total_preds.cpu().numpy()
        y_true = total_labels.cpu().numpy()

        if mode == 'test':
            self.predictions = list([data.label_list[idx] for idx in y_pred])
            self.true_labels = list([data.all_label_list[idx] for idx in y_true])
        else:
            self.predictions = list([data.label_list[idx] for idx in y_pred])
            self.true_labels = list([data.label_list[idx] for idx in y_true])

        if mode == 'eval':
            cm = confusion_matrix(y_true, y_pred)
            eval_score = F_measure(cm)['F1-score']
            print('acc', round(accuracy_score(y_true, y_pred) * 100, 2))
            return eval_score

        elif mode == 'test':
            unkdata_with_noise = "unkdataset_with_noise"
            if not os.path.exists(unkdata_with_noise):
                os.makedirs(unkdata_with_noise)

            unk_path = "unkdataset-{}-known_cls_ratio-{}-seed-{}-lr-{}.pkl".format(
                args.dataset, args.known_cls_ratio,
                args.seed, args.lr)
            unk_path = os.path.join(unkdata_with_noise, unk_path)
            with open(unk_path, 'wb') as fp:
                pickle.dump({'data': unkDataset}, fp)
            y_true = [item if item < data.unseen_token_id else data.unseen_token_id for item in y_true]
            cm = confusion_matrix(y_true, y_pred)
            results = F_measure(cm)
            acc = round(accuracy_score(y_true, y_pred) * 100, 2)
            results['Accuracy'] = acc

            self.test_results = results
            self.save_results(args)

    def train(self, args, data):

        criterion_boundary = BoundaryLoss(num_labels=data.num_labels, feat_dim=args.feat_dim)
        self.delta = F.softplus(criterion_boundary.delta)
        optimizer = torch.optim.Adam(criterion_boundary.parameters(), lr=args.lr_boundary)
        self.centroids = self.centroids_cal(args, data)

        wait = 0
        best_delta, best_centroids = None, None

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            for step, batch in enumerate(tqdm(data.train_dataloader, desc="Iteration")):
                batch.pop("label_text")
                batch = tuple(t.to(self.device) for _, t in batch.items())
                input_ids, input_mask, x_attention_mask, labels, label_ids = batch
                with torch.set_grad_enabled(True):
                    features = self.model.forward(input_ids=input_ids, attention_mask=input_mask,
                                                  labels=labels, x_attention_mask=x_attention_mask, label_ids=None)[
                        'feature']
                    loss, self.delta = criterion_boundary(features, self.centroids, label_ids)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    tr_loss += loss.item()

                    nb_tr_examples += input_ids.size(0)
                    nb_tr_steps += 1

            self.delta_points.append(self.delta)

            loss = tr_loss / nb_tr_steps
            print('train_loss', loss)

            eval_score = self.evaluation(args, data, mode="eval")
            print('eval_score', eval_score, wait)
            print("best_score", self.best_eval_score)

            if eval_score > self.best_eval_score:
                wait = 0
                self.best_eval_score = eval_score
                best_delta = self.delta
                best_centroids = self.centroids
            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.delta = best_delta
        self.centroids = best_centroids

    def class_count(self, labels):
        class_data_num = []
        for l in np.unique(labels):
            num = len(labels[labels == l])
            class_data_num.append(num)
        return class_data_num

    def centroids_cal(self, args, data):
        centroids = torch.zeros(data.num_labels, args.feat_dim).cuda()
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        with torch.set_grad_enabled(False):
            for batch in data.train_dataloader:
                batch.pop("label_text")
                batch = tuple(t.to(self.device) for _, t in batch.items())
                input_ids, input_mask, x_attention_mask, labels, label_ids = batch
                features = self.model(input_ids=input_ids, attention_mask=input_mask, labels=labels,
                                      x_attention_mask=x_attention_mask,
                                      label_ids=None)['feature']
                total_labels = torch.cat((total_labels, label_ids))
                for i in range(len(label_ids)):
                    label = label_ids[i]
                    centroids[label] += features[i]

        total_labels = total_labels.cpu().numpy()
        centroids /= torch.tensor(self.class_count(total_labels)).float().unsqueeze(1).cuda()

        return centroids

    def restore_model(self, args, device):
        # output_model_file = os.path.join(args.pretrain_dir, WEIGHTS_NAME)
        print("ReLoading Model Parameter")
        model_filepath = "model/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}".format(
            args.dataset, args.known_cls_ratio, args.seed, args.lr)
        self.model.load_state_dict(
            torch.load(os.path.join(model_filepath, "checkpoint.pkl"), map_location=self.device))

    def save_results(self, args):
        if not os.path.exists(args.save_results_path):
            os.makedirs(args.save_results_path)

        var = [args.dataset, args.known_cls_ratio, args.labeled_ratio, args.seed, args.lr]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed', 'lr']
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(self.test_results, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = '{}_adb_results.csv'.format(args.dataset)
        results_path = os.path.join(args.save_results_path, file_name)

        if not os.path.exists(results_path):
            ori = []
            ori.append(values)
            df1 = pd.DataFrame(ori, columns=keys)
            df1.to_csv(results_path, index=False)
        else:
            df1 = pd.read_csv(results_path)
            new = pd.DataFrame(results, index=[1])
            df1 = df1.append(new, ignore_index=True)
            df1.to_csv(results_path, index=False)
        data_diagram = pd.read_csv(results_path)

        print('test_results', data_diagram)


if __name__ == '__main__':
    args = load_parameters()
    set_seed(args.seed)

    datafile = "datafile/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}".format(
        args.dataset, args.known_cls_ratio, args.seed, args.lr)

    if not os.path.exists(datafile):
        os.makedirs(datafile)


    @cache_results("{}/data.pkl".format(datafile), _hash_param=False)
    def load_data(arg):
        pipe = Data(arg)
        return pipe


    data = load_data(args)
    print("train dataset length: {}\t dev dataset length: {}\t test dataset length: {}".format(
        len(data.train_dataset), len(data.eval_dataset), len(data.test_dataset)))
    train_dl = TorchDataLoader(data.train_dataset, batch_size=args.train_batch_size, shuffle=True)
    dev_dl = TorchDataLoader(data.eval_dataset, batch_size=args.eval_batch_size, shuffle=True)
    test_dl = TorchDataLoader(data.test_dataset, batch_size=args.eval_batch_size, shuffle=True)
    data.train_dataloader = train_dl
    data.eval_dataloader = dev_dl
    data.test_dataloader = test_dl
    args.num_labels = data.num_labels

    print('Pre-training finished!')

    manager = ModelManager(args, data, None)
    print('Training begin...')
    manager.train(args, data)
    # manager.generate_label(args, data, None)
    print('Training finished!')

    print('Evaluation begin...')
    manager.evaluation(args, data, mode="test")
    print('Evaluation finished!')
