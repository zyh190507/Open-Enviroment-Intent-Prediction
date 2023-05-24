#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys

sys.path.append("../")

import torch
from torch import nn
from transformers import AutoTokenizer
from T5_prefix_tuning_with_multi_label_v2.base import PushToHubFriendlyModel, T5ClassificationHead
from T5_prefix_tuning_with_multi_label_v2.modeling_t5 import T5ForConditionalGeneration


class MultiLabelCircleLoss(nn.Module):
    def __init__(self, reduction="mean", inf=1e12):
        """CircleLoss of MultiLabel, 多个目标类的多标签分类场景，希望“每个目标类得分都不小于每个非目标类的得分”
        多标签分类的交叉熵(softmax+crossentropy推广, N选K问题), LSE函数的梯度恰好是softmax函数
        让同类相似度与非同类相似度之间拉开一定的margin。
          - 使同类相似度比最大的非同类相似度更大。
          - 使最小的同类相似度比最大的非同类相似度更大。
          - 所有同类相似度都比所有非同类相似度更大。
        urls: [将“softmax+交叉熵”推广到多标签分类问题](https://spaces.ac.cn/archives/7359)
        args:
            reduction: str, Specifies the reduction to apply to the output, 输出形式.
                            eg.``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``
            inf: float, Minimum of maths, 无穷大.  eg. 1e12
        returns:
            Tensor of loss.
        examples:
            >>> label, logits = [[1, 1, 1, 1], [0, 0, 0, 1]], [[0, 1, 1, 0], [1, 0, 0, 1],]
            >>> label, logits = torch.tensor(label).float(), torch.tensor(logits).float()
            >>> loss = MultiLabelCircleLoss()(logits, label)
        """
        super(MultiLabelCircleLoss, self).__init__()
        self.reduction = reduction
        self.inf = inf  # 无穷大

    def forward(self, logits, labels):
        logits = (1 - 2 * labels) * logits              # <3, 4>
        logits_neg = logits - labels * self.inf         # <3, 4>
        logits_pos = logits - (1 - labels) * self.inf   # <3, 4>
        zeros = torch.zeros_like(logits[..., :1])       # <3, 1>
        logits_neg = torch.cat([logits_neg, zeros], dim=-1)  # <3, 5>
        logits_pos = torch.cat([logits_pos, zeros], dim=-1)  # <3, 5>
        neg_loss = torch.logsumexp(logits_neg, dim=-1)       # <3, >
        pos_loss = torch.logsumexp(logits_pos, dim=-1)       # <3, >
        loss = neg_loss + pos_loss
        if "mean" == self.reduction:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class Model(PushToHubFriendlyModel):
    def __init__(self, args):
        super().__init__()
        self.args = args

        """The prefix-tuning code"""

        self.preseqlen = args.pre_seq_len
        self.mid_dim = args.prefix_hidden_size

        print("prefix-tuning sequence length is {}.".format(self.preseqlen))

        # Load tokenizer and model.
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
        self.pretrain_model = T5ForConditionalGeneration.from_pretrained(
            args.model_name
        )
        self.config = self.pretrain_model.config
        if isinstance(self.pretrain_model, T5ForConditionalGeneration):
            self.match_n_layer = self.config.num_decoder_layers
            self.match_n_head = self.config.num_heads
            self.n_embd = self.config.d_model
            self.match_n_embd = self.config.d_kv
        else:
            raise ValueError("Other models are not supported yet!")

        # Prefix related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())

        self.wte = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )

        self.wte_enc = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_enc = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        self.wte_dec = nn.Embedding(self.preseqlen, self.n_embd)
        self.control_trans_dec = nn.Sequential(
            nn.Linear(self.n_embd, self.mid_dim),
            nn.Tanh(),
            nn.Linear(self.mid_dim, self.match_n_layer * 2 * self.match_n_head * self.match_n_embd),
        )
        self.dropout = nn.Dropout(args.prefix_dropout)

        for name, param in self.pretrain_model.named_parameters():
            param.requires_grad = False
        # 分类头
        self.transfomer_layer = torch.nn.TransformerEncoderLayer(d_model=self.n_embd,
                                                                 nhead=self.match_n_head, batch_first=True)
        self.classifer = T5ClassificationHead(input_dim=self.n_embd, inner_dim=self.n_embd,
                                              num_classes=args.num_labels, pooler_dropout=self.config.dropout_rate)

        self.lamda_loss = args.lamda_loss
        self.m_loss = args.m_loss

        total_mem = 0
        for param in self.parameters():
            if param.requires_grad:
                total_mem += param.numel()

        t5_mem = 0
        for param in self.pretrain_model.parameters():
            t5_mem += param.numel()

        print("total mem: {}\tT5 mem: {}".format(total_mem, t5_mem))

    def get_prompt(self, bsz=None, sample_size=1):
        old_bsz = bsz
        bsz = bsz * sample_size
        input_tokens = self.input_tokens.unsqueeze(0).expand(bsz, -1)
        temp_control = self.wte(input_tokens)

        past_key_values = self.control_trans(temp_control)  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        # Cross prefix
        temp_control_dec = self.wte_dec(input_tokens)
        past_key_values_dec = self.control_trans_dec(
            temp_control_dec
        )  # bsz, seqlen, layer*emb

        bsz, seqlen, _ = past_key_values_dec.shape
        past_key_values_dec = past_key_values_dec.view(
            bsz, seqlen, self.match_n_layer * 2, self.match_n_head, self.match_n_embd
        )
        past_key_values_dec = self.dropout(past_key_values_dec)
        past_key_values_dec = past_key_values_dec.permute([2, 0, 3, 1, 4]).split(2)

        # Encoder prefix
        input_tokens_enc = (
            self.input_tokens.unsqueeze(0).expand(old_bsz, -1)
        )
        temp_control_enc = self.wte_enc(input_tokens_enc)

        past_key_values_enc = self.control_trans_enc(
            temp_control_enc
        )  # bsz, seqlen, layer*emb

        bsz_enc, seqlen, _ = past_key_values_enc.shape
        past_key_values_enc = past_key_values_enc.view(
            bsz_enc,
            seqlen,
            self.match_n_layer * 2,
            self.match_n_head,
            self.match_n_embd,
        )
        past_key_values_enc = self.dropout(past_key_values_enc)
        past_key_values_enc = past_key_values_enc.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for i, key_val in enumerate(past_key_values):
            temp = dict()
            temp["decoder_prompt"] = {
                "prev_key": key_val[0].contiguous(),
                "prev_value": key_val[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val.device)
                    .bool()
                # bsz, preseqlen
            }
            key_val_dec = past_key_values_dec[i]
            temp["cross_attention_prompt"] = {
                "prev_key": key_val_dec[0].contiguous(),
                "prev_value": key_val_dec[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz, seqlen)
                    .to(key_val_dec.device)
                    .bool(),
            }
            key_val_enc = past_key_values_enc[i]
            temp["encoder_prompt"] = {
                "prev_key": key_val_enc[0].contiguous(),
                "prev_value": key_val_enc[1].contiguous(),
                "prev_key_padding_mask": torch.zeros(bsz_enc, seqlen)
                    .to(key_val_enc.device)
                    .bool(),
            }
            result.append(temp)

        return result

    def forward(self,
                input_ids,
                attention_mask,
                labels,
                multi_labels=None,
                label_ids=None
                ):
        bsz = input_ids.shape[0]
        # 生成 prompt
        past_prompt = self.get_prompt(
            bsz=bsz,
        )
        outputs = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
            output_hidden_states=True,
        )
        # T5Encoder 的 last_hidden_state 来做分类, 先经过 transformer layer 再取平均
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        transformer_layers = self.transfomer_layer(encoder_last_hidden_state).mean(dim=1)
        cls_logits, cls_features = self.classifer(transformer_layers)

        # 计算联合损失
        gen_loss = outputs.loss
        total_loss = 0
        cls_loss = 0
        ce_loss = 0
        if multi_labels is not None:
            loss_fct = MultiLabelCircleLoss()
            ce_loss_fct = torch.nn.CrossEntropyLoss()
            ce_loss = ce_loss_fct(cls_logits, label_ids)
            cls_loss = loss_fct.forward(cls_logits, multi_labels)
            cls_loss_tl = cls_loss * self.m_loss + ce_loss * (1-self.m_loss)
            total_loss = gen_loss * self.lamda_loss + cls_loss_tl * (1-self.lamda_loss)

        return {'loss': total_loss, "feature": cls_features, 'logits': cls_logits, 'gen_loss': gen_loss,
                'cls_loss': cls_loss, 'ce_loss': ce_loss}

    def evaluate_step(self,
                      input_ids,
                      attention_mask,
                      labels=None,
                      ):
        if labels.shape[1] > 1:
            raise ValueError("error")
        # 生成 beam=3 的句子
        generate_tokens = self.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=labels,
                                        early_stopping=True,
                                        num_return_sequences=3,
                                        num_beams=3, output_scores=True, max_length=10,
                                        min_length=2,
                                        return_dict_in_generate=True,
                                        )
        # 获得 cls_logits 好计算分类的准确率
        result_dict = {}
        if labels is not None:
            # 得到 logits
            result_dict = self.forward(input_ids=input_ids, attention_mask=attention_mask,
                                       labels=labels)
        result_dict.update({'decoder_tokens': generate_tokens.sequences})

        return result_dict

    def generate(self,
                 input_ids,
                 attention_mask,
                 **kwargs):

        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams']
        )

        generated_ids = self.pretrain_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids
