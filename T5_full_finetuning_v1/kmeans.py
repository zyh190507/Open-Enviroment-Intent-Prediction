import os
import sys

sys.path.append("../")

import torch
import pickle
import functools
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score, adjusted_rand_score, accuracy_score, \
    rand_score, adjusted_mutual_info_score

from fastNLP import cache_results, TorchDataLoader

from T5_full_finetuning_v1.pipe import Data, set_seed
from T5_full_finetuning_v1.model import OODT5Dection
from T5_full_finetuning_v1.utils import load_parameters


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
    acc = sum([w[i, j] for i, j in ind]) / y_pred.size
    return acc


def gen_acc_score(y_true: list, y_pred: list, ind_ood_idx_threshold):
    assert len(y_true) == len(y_pred)
    n_sample = len(y_true)
    # 聚类然后扔掉长度 1 的类
    y_true_cluster = [[] for _ in range(len(np.unique(y_pred)))]
    y_pred_cluster = [[] for _ in range(len(np.unique(y_pred)))]
    for clust, y_label in zip(y_pred, y_true):
        y_true_cluster[clust].append(y_label)
        y_pred_cluster[clust].append(clust)

    y_pred, y_true = [], []
    for ind, clust in enumerate(y_pred_cluster):
        if len(clust) > 1:
            y_true.append(y_true_cluster[ind])
            y_pred.append(y_pred_cluster[ind])
    y_pred = functools.reduce(lambda x, y: x + y, y_pred)
    y_true = functools.reduce(lambda x, y: x + y, y_true)


    y_true_clean = []
    y_pred_clean = []

    # 取出 ood 中 ind 的标签， 判别全为错
    for pred_el, true_el in zip(y_pred, y_true):
        if true_el >= ind_ood_idx_threshold:
            y_pred_clean.append(pred_el)
            y_true_clean.append(true_el - ind_ood_idx_threshold)
    assert len(y_true_clean) == len(y_pred_clean)

    y_true_clean = np.array(y_true_clean)
    y_pred_clean = np.array(y_pred_clean)
    D1 = max(y_pred_clean) + 1
    D2 = max(y_true_clean) + 1
    # 计算权重分配矩阵 （D1， D2）
    w = np.zeros((D1, D2))
    for i in range(y_true_clean.size):
        w[y_pred_clean[i], y_true_clean[i]] += 1
    # 匈牙利算法多次匹配对应的标签，然后将对应位置[row, col]的值就是做的正确的值
    correct = 0
    while True:
        # 最大匹配
        row_ind, col_ind = linear_sum_assignment(w, True)
        for row_id, col_id in zip(row_ind, col_ind):
            correct += w[row_id, col_id]
        # 删除匹配好的行
        w = np.delete(w, row_ind, axis=0)
        if w.shape[0] == 0:
            break
    acc = round((correct / n_sample * 100), 2)
    return acc


def clustering_score(y_true, y_pred):
    return {
        'ACC': round(clustering_accuracy_score(y_true, y_pred) * 100, 2),
        'ARI': round(adjusted_rand_score(y_true, y_pred) * 100, 2),
        'NMI': round(normalized_mutual_info_score(y_true, y_pred) * 100, 2),
        'ANMI': round(adjusted_mutual_info_score(y_true, y_pred) * 100, 2)
        # "Gen_ACC": round(gen_acc_score(y_true, y_pred, ind_ood_idx_threshold), 2)
    }


def save_results(arg, test_results):
    if not os.path.exists(arg.save_results_path):
        os.makedirs(arg.save_results_path)

    var = [arg.dataset, arg.known_cls_ratio, arg.labeled_ratio, arg.seed]
    names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed']
    vars_dict = {k: v for k, v in zip(names, var)}
    results = dict(test_results, **vars_dict)
    keys = list(results.keys())
    values = list(results.values())

    file_name = 'kmeans_results.csv'
    results_path = os.path.join(arg.save_results_path, file_name)

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


def ood_of_ind_ratio_(y_true, ind_ood_idx_threshold: int):
    total = len(y_true)
    error = 0
    for item in y_true:
        if item < ind_ood_idx_threshold:
            error += 1
    return error / total


if __name__ == '__main__':
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


    set_seed(args.seed)
    data = load_data(args)
    args.num_labels = data.num_labels

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # unkDs = get_unkdataset_of_dataset(data.test_dataset, data.unseen_token_id)
    unkDs = "unkdataset_with_noise/unkdataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-train_batch_size-{}.pkl".format(
        args.dataset, args.known_cls_ratio, args.seed, args.lr, args.train_batch_size)
    with open(unkDs, "rb") as fp:
        unkDs = pickle.load(fp)['data']
    # dl = TorchDataLoader(data.test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    model = OODT5Dection(args)
    model_filepath = "model/dataset-{}-known_cls_ratio-{}-seed-{}-lr_{}-train_batch_size-{}".format(
        args.dataset, args.known_cls_ratio, args.seed, args.lr, args.train_batch_size)
    model.load_state_dict(
        torch.load(os.path.join(model_filepath, "checkpoint.pkl"), map_location=device))
    model.to(device)
    dl = TorchDataLoader(unkDs, batch_size=32, shuffle=False)
    label_list = []
    sent_feat = []
    for batch in tqdm(dl, desc="Iteration"):
        input_ids, input_mask, label_ids, labels = batch['input_ids'].to(device), batch[
            'attention_mask'].to(device), \
                                                   batch['label_ids'].to(device), batch['labels'].to(
            device)
        # 收集真实标签
        label_list.extend(label_ids.cpu().numpy())
        # 得到 T5Enocder 的最后一层的句子的表示
        with torch.set_grad_enabled(False):
            pooled_output = model(input_ids=input_ids, attention_mask=input_mask, labels=labels,
                                  label_ids=None)['encoder_hidden']
            sent_feat.append(pooled_output.cpu())
    #  将所有批次的句子表示拼接在一起
    sent_feat = torch.cat(sent_feat, dim=0)
    y_true = np.array(label_list)
    cluster_num = len(data.all_label_list) * 3
    km = KMeans(n_clusters=cluster_num).fit(sent_feat)
    y_pred = km.labels_

    print("Cluster {}".format(len(np.unique(y_pred))))
    y_cluster = [0 for _ in range(len(np.unique(y_pred)))]
    y_cluster_list = [[] for _ in range(len(np.unique(y_pred)))]
    for item, y_label in zip(y_pred, y_true):
        y_cluster[item] += 1
        y_cluster_list[item].append(y_label)
    print(np.array(y_cluster).T)
    print(np.array(y_cluster_list))
    results = clustering_score(y_true, y_pred)
    print('results', results)

    print(ood_of_ind_ratio_(y_true, data.unseen_token_id))
    save_results(args, results)
