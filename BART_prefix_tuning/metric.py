import rouge
from fastNLP import Metric
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F


class RougeMetric(Metric):

    def __init__(self, tokenizer):
        super(RougeMetric, self).__init__()
        self.score = []
        self.tokenizer = tokenizer
        self.rouge = rouge.Rouge()

    def get_metric(self) -> dict:
        score = sum(self.score) / len(self.score)
        self.score = []
        return {'rouge_score': score}

    def update(self, label_text, decoder_tokens):
        decoded_preds = self.tokenizer.batch_decode(decoder_tokens, skip_special_tokens=True)
        # print(label_text)
        descript = []
        num_beams = []
        for beam_pred in decoded_preds:
            num_beams.append(beam_pred.strip())
            if len(num_beams) == 3:
                descript.append("; ".join(num_beams))
                num_beams = []
        # print(descript)
        score = self.rouge.get_scores(hyps=descript, refs=label_text, avg=True)
        self.score.append(score['rouge-1']['f'] * 0.2 + score['rouge-2']['f'] * 0.5 + score['rouge-l']['f'] * 0.3)


class Accuracy(Metric):

    def __init__(self, backend='auto', aggregate_when_get_metric: bool = None):
        super(Accuracy, self).__init__(backend=backend, aggregate_when_get_metric=aggregate_when_get_metric)
        self.register_element(name='correct', value=0, aggregate_method='sum', backend=backend)
        self.register_element(name='total', value=0, aggregate_method="sum", backend=backend)

    def get_metric(self) -> dict:
        evaluate_result = {'acc': round(self.correct.get_scalar() / (self.total.get_scalar() + 1e-12), 6)}
        return evaluate_result

    def update(self, logits, label_ids):
        _, logits = torch.nn.Softmax(dim=-1)(logits).max(dim=-1)
        self.correct += (logits == label_ids).sum().item()
        self.total += len(label_ids)


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits


class BoundaryLoss(nn.Module):

    def __init__(self, num_labels=10, feat_dim=2):
        super(BoundaryLoss, self).__init__()
        self.num_labels = num_labels
        self.feat_dim = feat_dim
        self.delta = nn.Parameter(torch.randn(num_labels).cuda())
        nn.init.normal_(self.delta)

    def forward(self, pooled_output, centroids, labels):
        logits = euclidean_metric(pooled_output, centroids)
        probs, preds = F.softmax(logits.detach(), dim=1).max(dim=1)
        delta = F.softplus(self.delta)
        c = centroids[labels]
        d = delta[labels]
        x = pooled_output

        euc_dis = torch.norm(x - c, 2, 1).view(-1)
        pos_mask = (euc_dis > d).type(torch.cuda.FloatTensor)
        neg_mask = (euc_dis < d).type(torch.cuda.FloatTensor)

        pos_loss = (euc_dis - d) * pos_mask
        neg_loss = (d - euc_dis) * neg_mask
        loss = pos_loss.mean() + neg_loss.mean()

        return loss, delta


def F_measure(cm):
    idx = 0
    rs, ps, fs = [], [], []
    n_class = cm.shape[0]

    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        rs.append(r * 100)
        ps.append(p * 100)
        fs.append(f * 100)

    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)
    result = {}
    result['Known'] = f_seen
    result['Open'] = f_unseen
    result['F1-score'] = f

    return result
