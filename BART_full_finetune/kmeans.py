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

from BART_full_finetune.pipe import Data, set_seed
from BART_full_finetune.model import Model
from BART_full_finetune.utils import load_parameters


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


def clustering_score(y_true, y_pred):
    return {
        'ACC': round(clustering_accuracy_score(y_true, y_pred) * 100, 2),
        'ARI': round(adjusted_rand_score(y_true, y_pred) * 100, 2),
        'NMI': round(normalized_mutual_info_score(y_true, y_pred) * 100, 2),
        'AMI': round(adjusted_mutual_info_score(y_true, y_pred) * 100, 2)
        # "Gen_ACC": round(gen_acc_score(y_true, y_pred, ind_ood_idx_threshold), 2)
    }


def save_results(arg, test_results):
    if not os.path.exists(arg.save_results_path):
        os.makedirs(arg.save_results_path)

    var = [arg.dataset, arg.known_cls_ratio, arg.labeled_ratio, arg.seed, args.delta]
    names = ['dataset', 'known_cls_ratio', 'labeled_ratio', 'seed', 'delta']
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

    datafile = "datafile/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-lmd-{}".format(
        args.dataset, args.known_cls_ratio,
        args.seed, args.lr, args.lamda_loss)

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
    if args.dataset == 'clinc':
        unkDs = "unkdataset_with_noise/unkdataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-lmd-{}-delta-{}.pkl".format(
                args.dataset, args.known_cls_ratio, args.seed, args.lr, args.lamda_loss, args.delta)
    else:
        unkDs = "unkdataset_with_noise/unkdataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-train_batch_size-{}.pkl".format(
            args.dataset, args.known_cls_ratio,
            args.seed, args.lr, args.train_batch_size)
    with open(unkDs, "rb") as fp:
        unkDs = pickle.load(fp)['data']
    # dl = TorchDataLoader(data.test_dataset, batch_size=args.eval_batch_size, shuffle=False)
    model = Model(args)
    model_filepath = "model/dataset-{}-known_cls_ratio-{}-seed-{}-lr-{}-lmd-{}".format(
        args.dataset, args.known_cls_ratio,
        args.seed, args.lr, args.lamda_loss)
    model.load_state_dict(
        torch.load(os.path.join(model_filepath, "checkpoint.pkl"), map_location=device))
    model.to(device)
    dl = TorchDataLoader(unkDs, batch_size=32, shuffle=False)
    label_list = []
    sent_feat = []
    for batch in tqdm(dl, desc="Iteration"):
        batch = tuple(t.to(device) for _, t in batch.items())
        input_ids, input_mask, label_ids, labels = batch
        # 收集真实标签
        label_list.extend(label_ids.cpu().numpy())
        # 得到 T5Enocder 的最后一层的句子的表示
        with torch.set_grad_enabled(False):
            pooled_output = model.forward(input_ids=input_ids, attention_mask=input_mask,
                                          labels=labels, label_ids=None)['sent_emb']
            sent_feat.append(pooled_output.cpu())
    #  将所有批次的句子表示拼接在一起
    sent_feat = torch.cat(sent_feat, dim=0)
    y_true = np.array(label_list)
    # cluster_num = min(len(data.all_label_list) * 3, len(label_list) // 2)
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
