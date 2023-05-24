import sys
import os

sys.path.append("../")

import torch
from transformers import get_linear_schedule_with_warmup
from fastNLP import cache_results, Trainer, Event, TorchDataLoader, EarlyStopCallback, LRSchedCallback, RandomSampler

from T5_full_finetuning_v1.pipe import Data, set_seed
from T5_full_finetuning_v1.model import OODT5Dection
from T5_full_finetuning_v1.utils import load_parameters
from T5_full_finetuning_v1.metric import RougeMetric, TextAccuracy, Accuracy

best = 0


@Trainer.on(Event.on_evaluate_end())
def save_model(trainer, result):
    global best, args
    model_filepath = "model/dataset-{}-known_cls_ratio-{}-seed-{}-lr_{}-train_batch_size-{}".format(
        args.dataset, args.known_cls_ratio, args.seed, args.lr, args.train_batch_size)

    if best < result['F1-score#acc']:
        if not os.path.exists(model_filepath):
            os.makedirs(model_filepath)
        model_name = "checkpoint.pkl"
        model_file = os.path.join(model_filepath, model_name)
        torch.save(trainer.model.state_dict(), model_file)
        best = result['F1-score#acc']


if __name__ == '__main__':
    print('Data and Parameters Initialization...')
    args = load_parameters()
    set_seed(args.seed)

    datafile = "datafile/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-train_batch_size-{}".format(
        args.dataset, args.known_cls_ratio, args.seed, args.lr, args.train_batch_size)

    if not os.path.exists(datafile):
        os.makedirs(datafile)


    @cache_results("{}/data.pkl".format(datafile), _hash_param=False)
    def load_data(arg):
        pipe = Data(arg)
        return pipe


    data = load_data(args)
    print("train dataset length: {}\t dev dataset length: {}\t test dataset length: {}".format(
        len(data.train_dataset), len(data.eval_dataset), len(data.test_dataset)))
    train_dl = TorchDataLoader(data.train_dataset, batch_size=args.train_batch_size,
                               sampler=RandomSampler(dataset=data.train_dataset,
                                                     shuffle=True, seed=args.seed))
    dev_dl = TorchDataLoader(data.eval_dataset, batch_size=args.eval_batch_size,
                             sampler=RandomSampler(dataset=data.eval_dataset,
                                                   shuffle=True, seed=args.seed))
    test_dl = TorchDataLoader(data.test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    print('Pre-training begining!')
    # train ind pretrained parameter
    args.num_labels = data.num_labels
    model = OODT5Dection(args)

    optim = torch.optim.AdamW(params=model.parameters(), lr=args.lr, eps=1e-8)
    callbacks = [
        EarlyStopCallback(monitor='F1-score#accMetric', patience=args.wait_patient),
        LRSchedCallback(scheduler=get_linear_schedule_with_warmup(optim,
                                                                  num_warmup_steps=50, num_training_steps=len(
                train_dl) * args.num_train_epochs))
    ]
    metric = RougeMetric(tokenizer=data.tokenizer)
    accmetric = TextAccuracy(tokenizer=data.tokenizer, data=data)
    accDectection = Accuracy()
    # metric = BM25Metric(tokenizer=data.tokenizer)
    train = Trainer(
        model=model,
        device=int(args.gpu_id),
        n_epochs=args.num_train_epochs,
        train_dataloader=train_dl,
        evaluate_dataloaders=dev_dl,
        optimizers=optim,
        driver='auto',
        metrics={'rouge': metric, "accMetric": accmetric, "accDectection": accDectection},
        callbacks=callbacks,
        # accumulation_steps=32,
    )
    train.run()

    print('Pre-training finished!')
