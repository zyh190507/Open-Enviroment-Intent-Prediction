import sys
import pickle
import os
import functools
from string import punctuation
from typing import List

sys.path.append("../")

import copy
import rama_py
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rouge import Rouge
from scipy.optimize import linear_sum_assignment
from fastNLP import DataSet, TorchDataLoader, cache_results
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, accuracy_score, \
    rand_score, adjusted_mutual_info_score

from T5_prefix_tuning_with_multi_label_v2.pipe import Pipe


def ood_of_ind_ratio_(y_true: List, ind_ood_idx_threshold: int):
    total = len(y_true)
    error = 0
    for item in y_true:
        if item < ind_ood_idx_threshold:
            error += 1
    return error / total


def hungray_aligment(y_true, y_pred):
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = np.transpose(np.asarray(linear_sum_assignment(w.max() - w)))
    return ind, w


def clustering_accuracy_score(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    y_true_clean = np.array(y_true)
    y_pred_clean = np.array(y_pred)

    ind, w = hungray_aligment(y_true_clean, y_pred_clean)
    acc = sum([w[i, j] for i, j in ind]) / len(y_pred)
    return acc


def clustering_score(y_true, y_pred, ind_ood_idx_threshold):
    return {
        'ACC': round(clustering_accuracy_score(y_true, y_pred) * 100, 2),
        # 'ACC': gen_acc_score(y_true, y_pred, ind_ood_idx_threshold),
        'ARI': round(adjusted_rand_score(y_true, y_pred) * 100, 2),
        "RI": round(rand_score(y_true, y_pred) * 100, 2),
        'NMI': round(normalized_mutual_info_score(y_true, y_pred) * 100, 2),
        'AMI': round(adjusted_mutual_info_score(y_true, y_pred) * 100, 2),
        "IND_Ratio": round(ood_of_ind_ratio_(y_true, ind_ood_idx_threshold) * 100, 2)}


class GenerateOOD:

    def __init__(self, args, data: Pipe, unkdataset_or_filename, pretrain_model_or_path, tokenizer_or_path,
                 rama_cluster_times: int = 1):
        self.args = args
        self.data = data
        self.p_node = args.p_node
        self.rama_cluster_times = rama_cluster_times + 1

        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(unkdataset_or_filename, str):
            print("Loading dataset with noise!")
            with open(unkdataset_or_filename, "rb") as fp:
                self.unkDs = pickle.load(fp)['data']
        elif isinstance(unkdataset_or_filename, DataSet):
            self.unkDs = unkdataset_or_filename
        else:
            raise ValueError("parameter is error!")

        if isinstance(pretrain_model_or_path, str):
            self.model = AutoModelForCausalLM.from_pretrained(pretrain_model_or_path).to(self.device)
        else:
            print("Loading Model parameter")
            self.model = pretrain_model_or_path.to(self.device)
            model_filepath = "model/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-pre_seq_len-{}-lmd-{}-m_loss-{}".format(
                args.dataset, args.known_cls_ratio,
                args.seed, args.lr, args.pre_seq_len, args.lamda_loss, args.m_loss)
            self.model.load_state_dict(
                torch.load(os.path.join(model_filepath, "checkpoint.pkl"), map_location=self.device))

        if isinstance(tokenizer_or_path, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_path)
        else:
            self.tokenizer = tokenizer_or_path

    def evaluation_ood(self):
        pred_description, label_list = self._generate_utterance()
        # y_pred = self._cluster_ood(pred_description)
        y_pred, y_true = self._cluster_ood_again(pred_description, label_list)
        result = clustering_score(y_true, y_pred, self.data.unseen_token_id)
        self.save_results(result)
        print(result)
        # print("Cluster {}".format(len(np.unique(y_pred))))
        # y_cluster = [0 for _ in range(len(np.unique(y_pred)))]
        # for data in y_pred:
        #     y_cluster[data] += 1
        # print(np.array(y_cluster).T)

    def _cluster_ood_again(self, pred_description, label_list):
        cluster_succ, cluster_label = [], []
        avg_sample_numb = len(pred_description) // len(np.unique(label_list))
        for i in range(self.rama_cluster_times):
            cluster_idx_offset = len(cluster_succ)
            cluster_list = self._cluster_ood(pred_description)
            # 分出簇来
            cluster_all, cluster_label_all, cluster_desp = [[] for _ in range(len(np.unique(cluster_list)))], \
                                                           [[] for _ in range(len(np.unique(cluster_list)))], \
                                                           [[] for _ in range(len(np.unique(cluster_list)))]
            node_map = {cl_num: idx for idx, cl_num in enumerate(np.unique(cluster_list))}
            assert len(cluster_list) == len(label_list) and len(label_list) == len(pred_description)
            for cluster_node, label_node, descrip in zip(cluster_list, label_list, pred_description):
                cluster_all[node_map[cluster_node]].append(cluster_node + cluster_idx_offset)
                cluster_label_all[node_map[cluster_node]].append(label_node)
                cluster_desp[node_map[cluster_node]].append(descrip)
            # 聚类次数够了
            if self.rama_cluster_times - 1 == i:
                cluster_succ.extend(cluster_all)
                cluster_label.extend(cluster_label_all)
                # 多次聚类的类别标签连续化
                re_cluster_succ = []
                for reIndex in range(len(cluster_succ)):
                    re_cluster_succ.append([reIndex] * len(cluster_succ[reIndex]))
                cluster_succ = re_cluster_succ
                break
            # 筛选出聚类效果不好的簇
            other_cluster_desp, other_cluster_label_all = [], []
            cluster_len = [len(_) for _ in cluster_all]
            for idx, num in enumerate(cluster_len):
                # 如果小于平均样本量三倍且数量大于 5 ，那么就是有效的簇; 否则收集起来再次分类
                if 5 < num < avg_sample_numb * 1.5:
                    cluster_succ.append(cluster_all[idx])
                    cluster_label.append(cluster_label_all[idx])
                else:
                    other_cluster_desp.append(cluster_desp[idx])
                    other_cluster_label_all.append(cluster_label_all[idx])
            if len(other_cluster_desp) == 0:
                break
            # 剩下的所有再进行一次聚类
            pred_description = functools.reduce(lambda x, y: x + y, other_cluster_desp)
            label_list = functools.reduce(lambda x, y: x + y, other_cluster_label_all)
            # self.p_node -= 0.01
        return functools.reduce(lambda x, y: x + y, cluster_succ), functools.reduce(lambda x, y: x + y, cluster_label)

    def _cluster_ood(self, pred_description):
        pred_rouge = []
        rouge = Rouge()
        # 将 topk 的 label 拼接成为 description 并计算 rouge
        for idx, descrip in tqdm(enumerate(pred_description)):
            # 一一计算 rouge
            descrip = [descrip] * len(pred_description)
            group_rouge = []
            for idy, rouge_out in enumerate(rouge.get_scores(descrip, pred_description)):
                # 保存为 （score, x, y）
                if idy > idx:
                    group_rouge.append(((rouge_out['rouge-1']['f'] + rouge_out['rouge-2']['f']
                                         + rouge_out['rouge-l']['f']) / 3, idx, idy))
            # 按照 score 排序
            group_rouge = sorted(group_rouge, key=lambda x: x[0], reverse=True)
            pred_rouge.append(group_rouge)
            # print(group_rouge)
        # 获取所有的边并随机去掉若干部分
        all_edge = functools.reduce(lambda x, y: x + y, pred_rouge)
        all_edge_non_zero = list(filter(lambda x: x[0] > 0, all_edge))
        # 将所有的边按照 score 排序
        all_edge_non_zero = sorted(all_edge_non_zero, key=lambda x: x[0], reverse=True)
        # 随机扔掉后面 20% 的有效边
        # all_edge_non_zero = all_edge_non_zero[:int(len(all_edge_non_zero) * 0.6)]
        threshold_edge_ind = int(len(all_edge_non_zero) * self.p_node)
        threshold_edge_score = all_edge_non_zero[threshold_edge_ind][0]
        rama_all_edge_non_zero = []
        for idx, edge in enumerate(all_edge_non_zero):
            if idx <= threshold_edge_ind:
                rama_all_edge_non_zero.append(edge)
            else:
                rama_all_edge_non_zero.append((edge[0] - threshold_edge_score, edge[1], edge[2]))

        rama_row, rama_ind, rama_score = [], [], []
        for (score, idx, idy) in rama_all_edge_non_zero:
            rama_row.append(idx)
            rama_ind.append(idy)
            rama_score.append(score)

        opts = rama_py.multicut_solver_options("PD+")
        res = rama_py.rama_cuda(rama_row, rama_ind, rama_score, opts)
        return res[0]

    def _generate_utterance(self):
        file_name = "GenerateOODData"
        arg = self.args
        pickle_name = "dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-pre_seq_len-{}-lmd-{}-m_loss-{}".format(
            arg.dataset, arg.known_cls_ratio,
            arg.seed, arg.lr, arg.pre_seq_len, args.lamda_loss, args.m_loss)
        file_path = os.path.join(file_name, pickle_name)
        if not os.path.exists(file_path):
            os.makedirs(file_path)

        # 尝试读取保存的生成文件， 若没有则去生成，否则可以重复使用
        if os.path.exists(os.path.join(file_path, "data.pkl")):
            print("Loading data for cache file!")
            with open(os.path.join(file_path, "data.pkl"), "rb") as fp:
                pred_description = pickle.load(fp)
                label_list = pickle.load(fp)
        else:
            # bad_word_id
            # bad_word_ids = self._bad_word_ids()
            label_list, num_beams, pred_description = [], [], []
            idx = 0
            # label_map = {i: label for i, label in enumerate(self.data.all_label_list)}
            # df_gen = pd.DataFrame(columns=['input', 'gen3', 'y_label'])
            # 按照顺序取
            dl = TorchDataLoader(self.unkDs, batch_size=2, shuffle=False)

            for batch in tqdm(dl, desc="Iteration"):
                input_ids, input_mask, label_ids, labels = batch['input_ids'].to(self.device), batch[
                    'attention_mask'].to(self.device), \
                                                           batch['label_ids'].to(self.device), batch['labels'].to(
                    self.device)
                # 收集真实标签
                label_list.extend(label_ids.cpu().numpy())
                generate_tokens = self.model.evaluate_step(input_ids=input_ids, attention_mask=input_mask,
                                                           labels=labels)

                tokens = generate_tokens['decoder_tokens']
                decoded_preds = self.tokenizer.batch_decode(tokens, skip_special_tokens=True)
                # decoded_input = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                idy = 0
                for pred_word in decoded_preds:
                    num_beams.append(pred_word)
                    if len(num_beams) == 3:
                        description = ", ".join(num_beams)
                        pred_description.append(description.strip())
                        num_beams = []
                        # df_gen = df_gen.append({'input': decoded_input[idy], 'gen3': description,
                        #                         'y_label': label_map[label_list[idx]],
                        #                         }, ignore_index=True)
                        idx += 1
                        idy += 1
                assert len(label_list) == len(pred_description)
                print(pred_description)
            # #  保存生成的文件
            # df_gen.to_csv("{}-lr-{}-pre_seq_len-{}-seed-{}_gen_result.csv".format(self.args.dataset, arg.lr,
            #                                                                       arg.pre_seq_len, args.seed))
            # 保存文件 pickle 文件
            print("Save data in cache files")
            with open(os.path.join(file_path, "data.pkl"), "wb") as fp:
                pickle.dump(pred_description, fp)
                pickle.dump(label_list, fp)

        return pred_description, label_list

    def save_results(self, result_dict):
        if not os.path.exists(self.args.save_results_path):
            os.makedirs(self.args.save_results_path)

        var = [self.args.dataset, self.args.known_cls_ratio, self.args.labeled_ratio, self.p_node,
               self.args.pre_seq_len, args.seed, args.prefix_dropout, self.rama_cluster_times,
               self.args.lr, self.args.train_batch_size, self.args.prefix_projection, self.args.prefix_hidden_size]
        names = ['dataset', 'known_cls_ratio', 'labeled_ratio', "node_save_ratio", "pre_seq_len", "seed",
                 "prefix_dropout", "rama_cluster_times", "lr", "train_batch_size", "prefix_projection",
                 "prefix_hidden_size"]
        vars_dict = {k: v for k, v in zip(names, var)}
        results = dict(result_dict, **vars_dict)
        keys = list(results.keys())
        values = list(results.values())

        file_name = '{}_results_gen.csv'.format(self.args.dataset)
        results_path = os.path.join(self.args.save_results_path, file_name)

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


if __name__ == '__main__':
    from T5_prefix_tuning_with_multi_label_v2.utils import load_parameters
    from T5_prefix_tuning_with_multi_label_v2.pipe import set_seed
    from T5_prefix_tuning_with_multi_label_v2.prefixtuning import Model
    from fastNLP import cache_results

    args = load_parameters()

    datafile = "datafile/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-pre_seq_len-{}-lmd-{}".format(
        args.dataset, args.known_cls_ratio,
        args.seed, args.lr, args.pre_seq_len, args.lamda_loss)
    if not os.path.exists(datafile):
        os.makedirs(datafile)


    @cache_results("{}/data.pkl".format(datafile), _hash_param=False)
    def load_data(arg):
        pipe = Pipe(arg)
        return pipe


    set_seed(args.seed)
    data = load_data(args)
    args.num_labels = len(data.known_multi_label_unique)
    # args.num_labels = data.num_labels

    unkDs = "unkdataset_with_noise/unkdataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-pre_seq_len-{}-lmd-{}.pkl".format(
                args.dataset, args.known_cls_ratio,
                args.seed, args.lr, args.pre_seq_len, args.lamda_loss)
    pretrain_model_path = Model(args)
    tokenizer_path = data.tokenizer
    args.p_node = 0.20
    # for rama_cluster_times in range(3):
    gen_ood = GenerateOOD(args, data, unkDs, pretrain_model_path, tokenizer_path, rama_cluster_times=4)
    gen_ood.evaluation_ood()
