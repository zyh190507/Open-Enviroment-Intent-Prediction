import os
import random
import numpy as np
import csv
import copy
import sys

import torch
from transformers import GPT2Tokenizer
from fastNLP import DataSet
from rouge import Rouge


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class Data:

    def __init__(self, args):
        set_seed(args.seed)
        max_seq_lengths = {'oos': 30, 'stackoverflow': 45, 'banking': 55, 'clinc': 30}
        args.max_seq_length = max_seq_lengths[args.dataset]
        self.max_seq_length = args.max_seq_length
        processor = DatasetProcessor()
        self.data_dir = os.path.join("../", args.data_dir, args.dataset)
        self.all_label_list = processor.get_labels(self.data_dir)
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))

        self.all_label_list_ = copy.deepcopy(self.known_label_list)
        for label in self.all_label_list:
            if label not in self.known_label_list:
                self.all_label_list_.append(label)
        self.all_label_list = self.all_label_list_

        assert self.known_label_list == self.all_label_list[:len(self.known_label_list)]

        self.num_labels = len(self.known_label_list)

        self.unseen_token = '<UNK>'

        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]
        self.train_examples = self.get_examples(processor, args, 'train')
        self.eval_examples = self.get_examples(processor, args, 'eval')
        self.test_examples = self.get_examples(processor, args, 'test')

        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        tokenizer.pad_token = tokenizer.eos_token
        self.tokenizer = tokenizer

        self.train_dataset = self.get_loader(self.train_examples, args, tokenizer, 'train')
        self.eval_dataset = self.get_loader(self.eval_examples, args, tokenizer, 'eval')
        self.test_dataset = self.get_loader(self.test_examples, args, tokenizer, 'test')

    def get_examples(self, processor, args, mode='train'):
        ori_examples = processor.get_examples(self.data_dir, mode)

        examples = []
        if mode == 'train':
            for example in ori_examples:
                if (example.label in self.known_label_list) and (np.random.uniform(0, 1) <= args.labeled_ratio):
                    examples.append(example)
        elif mode == 'eval':
            for example in ori_examples:
                if (example.label in self.known_label_list):
                    examples.append(example)
        elif mode == 'test':
            for example in ori_examples:
                examples.append(example)
        return examples

    def get_loader(self, examples, args, tokenizer, mode='train'):
        if mode == 'test':
            features = convert_examples_to_features(examples, self.all_label_list, args.max_seq_length, tokenizer,
                                                    mode)
        else:
            features = convert_examples_to_features(examples, self.label_list, args.max_seq_length, tokenizer,
                                                    mode)
        # features = convert_examples_to_features(examples, self.label_list, args.max_seq_length, tokenizer, mode)
        input_ids = [f.input_ids for f in features]
        input_mask = [f.attention_mask for f in features]
        label_ids = [f.label_id for f in features]
        labels = [f.labels for f in features]
        label_text = [f.label_text for f in features]
        x_attention_mask = [f.x_attention_mask for f in features]
        dataset = DataSet({'input_ids': input_ids, 'attention_mask': input_mask, 'label_text': label_text,
                           'x_attention_mask': x_attention_mask, 'labels': labels, 'label_ids': label_ids})

        return dataset


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, labels, label_id, label_text, x_attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.label_text = label_text
        self.labels = labels
        self.label_id = label_id
        self.x_attention_mask = x_attention_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self, data_dir):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines


class DatasetProcessor(DataProcessor):

    def get_examples(self, data_dir, mode):
        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'eval':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self, data_dir):
        """See base class."""
        import pandas as pd
        test = pd.read_csv(os.path.join(data_dir, "train.tsv"), sep="\t")
        labels = np.unique(np.array(test['label']))

        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            if len(line) != 2:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, mode):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    max_seq_length = 128
    for (ex_index, example) in enumerate(examples):
        clean_label = example.label.replace("_", " ").replace("-", " ")

        if mode == 'train':
            # 训练时候 pad 在右边即可， 输入为 X It was Y
            tokenizer.padding_side = 'right'
            template_text = "{} It was {}".format(example.text_a, clean_label)
            inputs = tokenizer(text=template_text, return_tensors='pt', padding='max_length',
                               truncation=True, max_length=max_seq_length)
            # 获取句子 X 的 mask 矩阵
            x_attention_mask = tokenizer(text=example.text_a, return_tensors='pt', padding='max_length',
                                         truncation=True, max_length=max_seq_length)['attention_mask']
            src_input_ids = inputs["input_ids"].squeeze(0)
            src_attention_mask = inputs["attention_mask"].squeeze(0)
            x_attention_mask = x_attention_mask.squeeze(0)
        else:
            # 测试生成 y 的时候，pad 需要在左边， 输入为 X It was
            tokenizer.padding_side = 'left'
            template_text = "{} It was".format(example.text_a)

            inputs_a = tokenizer(text=template_text, return_attention_mask=False,
                                 return_token_type_ids=False,
                                 )['input_ids']
            # 获取 x 以便得到 mask 矩阵
            x_inputs = tokenizer(text=example.text_a, return_attention_mask=False,
                                 return_token_type_ids=False,
                                 )['input_ids']
            # 计算得到 x 的 mask矩阵
            x_attention_mask = [1] * len(x_inputs) + [0] * (len(inputs_a)-len(x_inputs))
            src_attention_mask = [1] * len(inputs_a)
            # pad 到左边
            if len(inputs_a) < max_seq_length:
                inputs_a = [tokenizer.pad_token_id] * (max_seq_length-len(inputs_a)) + inputs_a
                src_attention_mask = [0] * (max_seq_length-len(src_attention_mask)) + src_attention_mask
                x_attention_mask = [0] * (max_seq_length-len(x_attention_mask)) + x_attention_mask

            x_attention_mask = torch.tensor(x_attention_mask)
            src_attention_mask = torch.tensor(src_attention_mask)
            src_input_ids = torch.tensor(inputs_a)

        # 获取监督信号 Y, 将 pad 位置变为 -100 以便忽略计算损失
        labels = copy.deepcopy(src_input_ids)
        labels[labels == tokenizer.pad_token_id] = -100

        label_id = label_map[example.label]
        label_text = clean_label
        if ex_index < 5:
            print("*** Example ***")
            print("guid: %s" % (mode))
            print("tokens: %s" % template_text)
            print("input_ids: %s" % " ".join([str(x) for x in src_input_ids.numpy()]))
            print("input_mask: %s" % " ".join([str(x) for x in src_attention_mask.numpy()]))
            print("x_input_mask: %s" % " ".join([str(x) for x in x_attention_mask.numpy()]))
            print("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=src_input_ids,
                          attention_mask=src_attention_mask,
                          x_attention_mask=x_attention_mask,
                          labels=labels,
                          label_id=label_id,
                          label_text=label_text,
                          ))
    return features
