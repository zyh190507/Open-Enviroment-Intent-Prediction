import argparse

import torch
from fastNLP import DataSet, Instance


def load_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='data', type=str,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument("--save_results_path", type=str, default='outputs', help="the path to save results")
    parser.add_argument("--pretrain_dir", default='models', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_name", default="facebook/bart-base", type=str,
                        help="The path for the pre-trained bert model.")
    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--dataset", default='banking', type=str,
                        help="The name of the dataset to train selected")
    parser.add_argument("--known_cls_ratio", default=0.75, type=float, help="The number of known classes")
    parser.add_argument("--labeled_ratio", default=1.0, type=float,
                        help="The ratio of labeled samples in the training set")
    parser.add_argument('--seed', type=int, default=52, help="random seed for initialization")
    parser.add_argument("--gpu_id", type=str, default="4", help="Select the GPU id")
    parser.add_argument("--lr", default=2e-3, type=float,
                        help="The learning rate of BERT.")
    parser.add_argument("--num_train_epochs", default=100.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--train_batch_size", default=64, type=int,
                        help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int,
                        help="Batch size for evaluation.")
    parser.add_argument("--wait_patient", default=10, type=int,
                        help="Patient steps for Early Stop.")
    parser.add_argument("--feat_dim", default=768, type=int, help="The feature dimension.")
    parser.add_argument("--lr_boundary", default=0.05, type=float)

    parser.add_argument("--pre_seq_len", default=12, type=int, help="prompt seq length")
    parser.add_argument("--prefix_hidden_size", default=768, type=int, help="prefix hidden layer")
    parser.add_argument("--prefix_dropout", default=0.5, type=float)
    parser.add_argument("--lamda_loss", default=0.5, type=float)
    parser.add_argument("--m_loss", default=0.5, type=float)
    parser.add_argument("--p_node", default=0.4, type=float)
    parser.add_argument("--delta", default=1.0, type=float)

    args = parser.parse_args()
    return args


def get_unkdataset_of_dataset(dataset: DataSet, unk_label_id: int) -> DataSet:
    unkDs = DataSet()
    for ins in dataset:
        if ins['label_ids'] >= unk_label_id:
            unkDs.append(
                Instance(input_ids=ins['input_ids'],
                         attention_mask=ins['attention_mask'],
                         labels=ins['labels'],
                         label_ids=ins['label_ids'])
            )
    return unkDs


def pack_labels_batch(labels_batch, device=None):
    batch = [[] for _ in range(len(labels_batch[0]))]
    for label_batch in labels_batch:
        for idx, label in enumerate(label_batch):
            batch[idx].append(label)
    if device:
        batch = [torch.stack(item, dim=0).to(device) for item in batch]
    else:
        batch = [torch.stack(item, dim=0) for item in batch]
    return batch
