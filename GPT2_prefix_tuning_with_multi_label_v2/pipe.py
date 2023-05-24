import os
import copy
import torch
import random
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer
from fastNLP import DataSet
from sklearn.preprocessing import MultiLabelBinarizer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_train_label_and_ds(root, dataset_name):
    filepath = os.path.join(root, "{}_expand_all_5.txt".format(dataset_name))
    example = []
    label_unique = []
    multi_label_unique_map = {}
    with open(filepath, "r", encoding="utf8") as fp:
        for idx, line in enumerate(fp):
            line = line.strip().strip("\n")
            line = line.split("###")
            text = line[0]
            multi_labels = line[1]
            # label 之间是 ","分割
            multi_labels = multi_labels.split(',')
            # 最后一个 label 是数据集原来的 label
            last_label = multi_labels.pop()
            label_unique.append(last_label)
            # 剩下5个 label
            if len(multi_labels) < 4:
                raise ValueError("index {} is error".format(idx))
            else:
                multi_labels = multi_labels[:4]

            # 每个真实标签对应的 multi label
            if last_label not in multi_label_unique_map:
                multi_label_unique_map[last_label] = copy.deepcopy(multi_labels)
            else:
                multi_label_unique_map[last_label].extend(multi_labels)

            example.append((text, multi_labels, last_label))
    # label 经过 set 处理
    label_unique = np.unique(label_unique)

    # 判断每个真实标签对应的4个标签是否相同
    multi_label_unique_map = {key: np.unique(value) for key, value in multi_label_unique_map.items()}
    for key, value in multi_label_unique_map.items():
        if len(value) == 4:
            continue
        raise ValueError(f"error")

    # example 中样本数据为(文本， 多个label(list), 原始的标签)
    return example, label_unique, multi_label_unique_map


def load_dev_test_ds(root, mode):
    if mode == 'dev':
        filepath = os.path.join(root, 'dev.tsv')
    elif mode == 'test':
        filepath = os.path.join(root, 'test.tsv')
    else:
        raise ValueError("mode: {} is nor exist".format(mode))
    example = []
    df = pd.read_csv(filepath, sep="\t", header=0)
    # 按照每行 读取
    for index, row in df.iterrows():
        example.append((row['text'], row['label']))
    return example


def filter_ds_by_ind_label(example, konw_label_list, mode):
    # 根据 ind 的 label 来筛选掉 ood 的数据
    # 只清洗 train, dev 而 test 数据不变
    new_example = []
    if mode == 'train':
        for tpl in example:
            if tpl[2] in konw_label_list:
                new_example.append(tpl)
    elif mode == 'dev':
        for tpl in example:
            if tpl[1] in konw_label_list:
                new_example.append(tpl)
    elif mode == 'test':
        for tpl in example:
            new_example.append(tpl)
    else:
        raise ValueError("mode: {} is nor exist".format(mode))
    return new_example


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer,
                                 known_multi_label_mapping, known_multi_label_unique, mode):
    """Loads a data file into a list of `InputBatch`s."""
    label_map = {}
    for i, label in enumerate(label_list):
        label_map[label] = i

    features = []
    # 得到多标签的 变换器
    mlb = MultiLabelBinarizer(classes=known_multi_label_unique)

    for (ex_index, example) in enumerate(examples):
        if mode == 'train':
            ## 训练的样本有多个标签增强
            text = example[0]
            multi_labels = example[1]
            last_label = example[2]

            # 训练时候 pad 在右边即可， 输入为 X It was Y
            tokenizer.padding_side = 'right'
            label_list = []
            src_inputs = []
            src_attention = []
            x_attention = []
            # 输入到 decoder
            for multi_label in multi_labels:
                clean_label = multi_label.replace("_", " ").replace("-", " ")
                template_text = "{} It was {}".format(text, clean_label)
                inputs = tokenizer(text=template_text, return_tensors='pt', padding='max_length',
                                   truncation=True, max_length=max_seq_length)
                # 获取句子 X 的 mask 矩阵
                x_attention_mask = tokenizer(text=text, return_tensors='pt', padding='max_length',
                                             truncation=True, max_length=max_seq_length)['attention_mask']
                src_input_ids = inputs["input_ids"].squeeze(0)
                src_attention_mask = inputs["attention_mask"].squeeze(0)
                x_attention_mask = x_attention_mask.squeeze(0)
                # 获取监督信号 Y, 将 pad 位置变为 -100 以便忽略计算损失
                labels = copy.deepcopy(src_input_ids)
                labels[labels == tokenizer.pad_token_id] = -100

                label_list.append(labels)
                src_inputs.append(src_input_ids)
                src_attention.append(src_attention_mask)
                x_attention.append(x_attention_mask)

            multi_label_hot = mlb.fit_transform([tuple(multi_labels)])[0]
            label_id = label_map[last_label]
            label_text = last_label.replace("_", " ").replace("-", " ")
            features.append({'input_ids': src_inputs, 'attention_mask': src_attention,
                             'multi_label_hot': multi_label_hot,
                             'label_list': label_list, 'label_id': label_id, 'label_text': label_text,
                             'x_attention_mask': x_attention})
            if ex_index < 5:
                print("*** Example ***")
                print("guid: %s" % (mode))
                print("tokens: %s" % template_text)
                # print("input_ids: %s" % " ".join([str(x) for x in src_input_ids.numpy()]))
                # print("input_mask: %s" % " ".join([str(x) for x in src_attention_mask.numpy()]))
                print("label: %s (id = %d)" % (last_label, label_id))
        elif mode == 'dev':
            text = example[0]
            last_label = example[1]
            multi_labels = known_multi_label_mapping[last_label]

            # 测试生成 y 的时候，pad 需要在左边， 输入为 X It was
            tokenizer.padding_side = 'left'
            template_text = "{} It was".format(text)
            inputs_a = tokenizer(text=template_text, return_attention_mask=False,
                                 return_token_type_ids=False,
                                 )['input_ids']
            # 获取 x 以便得到 mask 矩阵
            x_inputs = tokenizer(text=text, return_attention_mask=False,
                                 return_token_type_ids=False,
                                 )['input_ids']
            # 计算得到 x 的 mask矩阵
            x_attention_mask = [1] * len(x_inputs) + [0] * (len(inputs_a) - len(x_inputs))
            src_attention_mask = [1] * len(inputs_a)
            # pad 到左边
            if len(inputs_a) < max_seq_length:
                inputs_a = [tokenizer.pad_token_id] * (max_seq_length - len(inputs_a)) + inputs_a
                src_attention_mask = [0] * (max_seq_length - len(src_attention_mask)) + src_attention_mask
                x_attention_mask = [0] * (max_seq_length - len(x_attention_mask)) + x_attention_mask

            x_attention_mask = torch.tensor(x_attention_mask)
            src_attention_mask = torch.tensor(src_attention_mask)
            src_input_ids = torch.tensor(inputs_a)

            labels = copy.deepcopy(src_input_ids)
            labels[labels == tokenizer.pad_token_id] = -100

            label_id = label_map[last_label]
            label_text = last_label.replace("_", " ").replace("-", " ")
            multi_label_hot = mlb.fit_transform([tuple(multi_labels)])[0]

            features.append({'input_ids': [src_input_ids], 'attention_mask': [src_attention_mask],
                             'multi_label_hot': multi_label_hot,
                             'label_list': [labels], 'label_id': label_id, 'label_text': label_text,
                             'x_attention_mask': [x_attention_mask]})

            if ex_index < 5:
                print("*** Example ***")
                print("guid: %s" % (mode))
                print("tokens: %s" % text)
                print("output: %s" % template_text)
                print("input_ids: %s" % " ".join([str(x) for x in src_input_ids.numpy()]))
                print("input_mask: %s" % " ".join([str(x) for x in src_attention_mask.numpy()]))
                print("label: %s (id = %d)" % (last_label, label_id))
        else:
            text = example[0]
            last_label = example[1]

            # 测试生成 y 的时候，pad 需要在左边， 输入为 X It was
            tokenizer.padding_side = 'left'
            template_text = "{} It was".format(text)
            inputs_a = tokenizer(text=template_text, return_attention_mask=False,
                                 return_token_type_ids=False,
                                 )['input_ids']
            # 获取 x 以便得到 mask 矩阵
            x_inputs = tokenizer(text=text, return_attention_mask=False,
                                 return_token_type_ids=False,
                                 )['input_ids']
            # 计算得到 x 的 mask矩阵
            x_attention_mask = [1] * len(x_inputs) + [0] * (len(inputs_a) - len(x_inputs))
            src_attention_mask = [1] * len(inputs_a)
            # pad 到左边
            if len(inputs_a) < max_seq_length:
                inputs_a = [tokenizer.pad_token_id] * (max_seq_length - len(inputs_a)) + inputs_a
                src_attention_mask = [0] * (max_seq_length - len(src_attention_mask)) + src_attention_mask
                x_attention_mask = [0] * (max_seq_length - len(x_attention_mask)) + x_attention_mask

            x_attention_mask = torch.tensor(x_attention_mask)
            src_attention_mask = torch.tensor(src_attention_mask)
            src_input_ids = torch.tensor(inputs_a)

            labels = copy.deepcopy(src_input_ids)
            labels[labels == tokenizer.pad_token_id] = -100

            label_id = label_map[last_label]
            label_text = last_label.replace("_", " ").replace("-", " ")

            features.append({'input_ids': [src_input_ids], 'attention_mask': [src_attention_mask],
                             'multi_label_hot': label_id,
                             'label_list': [labels], 'label_id': label_id, 'label_text': label_text,
                             'x_attention_mask': [x_attention_mask]})

            if ex_index < 5:
                print("*** Example ***")
                print("guid: %s" % (mode))
                print("tokens: %s" % text)
                print("output: %s" % template_text)
                print("input_ids: %s" % " ".join([str(x) for x in src_input_ids.numpy()]))
                print("input_mask: %s" % " ".join([str(x) for x in src_attention_mask.numpy()]))
                print("label: %s (id = %d)" % (last_label, label_id))

    return features


class Pipe:

    def __init__(self, args):
        set_seed(args.seed)
        args.max_seq_length = 128
        self.args = args
        # 数据集的根目录
        data_dir = os.path.join("../", args.data_dir, args.dataset)

        # 加载多标签数据的 最后一个 label 作为真实的 label_map
        train_example, all_label_list, multi_label_unique_map = load_train_label_and_ds(data_dir, args.dataset)

        # 单标签对应的多标签 mapping -> {'label1': list1, 'label2': list2}
        self.multi_label_mapping = multi_label_unique_map

        # 划分出 ind 的 label
        self.all_label_list = all_label_list
        self.n_known_cls = round(len(self.all_label_list) * args.known_cls_ratio)
        self.known_label_list = list(np.random.choice(np.array(self.all_label_list), self.n_known_cls, replace=False))

        # 训练的多标签 mapping, 及训练的 多标签的 列表
        self.known_multi_label_mapping = {}
        known_multi_label_unique = []
        for label in self.known_label_list:
            self.known_multi_label_mapping[label] = self.multi_label_mapping[label]
            known_multi_label_unique.extend(self.multi_label_mapping[label])
        known_multi_label_unique = np.unique(known_multi_label_unique)
        self.known_multi_label_unique = known_multi_label_unique

        # 得到全部的 label
        self.all_label_list_ = copy.deepcopy(self.known_label_list)
        for label in self.all_label_list:
            if label not in self.known_label_list:
                self.all_label_list_.append(label)
        self.all_label_list = self.all_label_list_
        assert self.known_label_list == self.all_label_list[:len(self.known_label_list)]
        # 获取 dev， test 的 样本
        dev_example = load_dev_test_ds(data_dir, 'dev')
        test_example = load_dev_test_ds(data_dir, 'test')
        # 清洗掉 ood 数据
        self.train_example = filter_ds_by_ind_label(train_example, self.known_label_list, 'train')
        self.dev_example = filter_ds_by_ind_label(dev_example, self.known_label_list, 'dev')
        self.test_example = filter_ds_by_ind_label(test_example, self.known_label_list, 'test')

        self.tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加入 unk 标签 以便 dectect
        self.num_labels = len(self.known_label_list)
        self.unseen_token = '<UNK>'
        self.unseen_token_id = self.num_labels
        self.label_list = self.known_label_list + [self.unseen_token]

        self.train_dataset = self.get_loader(self.train_example, args, self.tokenizer, known_multi_label_unique,
                                             'train')
        self.eval_dataset = self.get_loader(self.dev_example, args, self.tokenizer, known_multi_label_unique, 'dev')
        self.test_dataset = self.get_loader(self.test_example, args, self.tokenizer, known_multi_label_unique, 'test')

    def get_loader(self, examples, args, tokenizer, known_multi_label_unique, mode='train'):
        if mode == 'test':
            features = convert_examples_to_features(examples, self.all_label_list, args.max_seq_length, tokenizer,
                                                    self.multi_label_mapping, known_multi_label_unique, mode)
        else:
            features = convert_examples_to_features(examples, self.label_list, args.max_seq_length, tokenizer,
                                                    self.known_multi_label_mapping, known_multi_label_unique, mode)

        input_ids = [f['input_ids'] for f in features]
        input_mask = [f['attention_mask'] for f in features]
        label_ids = [f['label_id'] for f in features]
        label_text = [f['label_text'] for f in features]
        label_list = [f['label_list'] for f in features]
        x_attention_mask = [f['x_attention_mask'] for f in features]
        multi_labels = [f['multi_label_hot'] for f in features]
        dataset = DataSet({'input_ids': input_ids, 'attention_mask': input_mask,
                           # "decoder_attention_mask": tg_attention_mask,
                           'multi_labels': multi_labels,
                           "x_attention_mask": x_attention_mask,
                           'label_ids': label_ids, "labels": label_list, "label_text": label_text})

        return dataset
