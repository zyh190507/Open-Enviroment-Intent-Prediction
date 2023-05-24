import sys

import torch
from torch import nn

from GPT2_prefix_tuning_with_multi_label_v2.modeling_gpt2 import GPT2LMHeadModel


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


class GPT2ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            num_classes: int,
            pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        # hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.nn.ReLU()(hidden_states)
        features = self.dropout(hidden_states)
        hidden_states = self.out_proj(features)
        return hidden_states, features


class Model(nn.Module):
    """Prefix tuning for GPT2 LM model"""

    def __init__(self, args):
        super().__init__()

        self.base_model = GPT2LMHeadModel.from_pretrained(args.model_name)
        config = self.base_model.config
        self.config = config
        # 给模型知道 pad 的值
        self.base_model.config.pad_token_id = self.base_model.config.eos_token_id
        # 冻结 GPT2 的参数
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.match_n_layer = config.n_layer
        self.match_n_head = config.n_head
        self.match_n_embd = config.n_embd // config.n_head
        self.n_embd = config.n_embd

        # Set up from config
        self.preseqlen = args.pre_seq_len
        self.prefix_hidden_size = args.prefix_hidden_size
        self.prefix_projection = args.prefix_projection

        # Prefix related.
        self.register_buffer('input_tokens', torch.arange(self.preseqlen).long())

        # 是否使用 2层 MLP 层
        if self.prefix_projection:
            self.wte_user = nn.Embedding(self.preseqlen, config.n_embd)
            self.control_trans_user = nn.Sequential(
                nn.Linear(config.n_embd, self.prefix_hidden_size),
                nn.Tanh(),
                nn.Linear(self.prefix_hidden_size, config.n_layer * 2 * config.n_embd)
            )
        else:
            self.wte_user = nn.Embedding(self.preseqlen, config.n_layer * 2 * config.n_embd)

        # Here we set prefix dropout prob = 0.4
        self.prefix_dropout = 0.4
        self.dropout = nn.Dropout(self.prefix_dropout)
        self.lamda_loss = args.lamda_loss
        self.m_loss = args.m_loss

        # 分类头
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=self.n_embd,
                                                                  nhead=self.match_n_head, batch_first=True)
        self.classifier = GPT2ClassificationHead(self.base_model.config.n_embd, self.base_model.config.n_embd,
                                                 args.num_labels, self.base_model.config.resid_pdrop)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
                total_param += param.numel()
        print('total param is {}'.format(total_param))

    def get_prompt(self, bsz=None, sample_size=1):
        bsz = bsz * sample_size
        input_tokens_user = self.input_tokens.unsqueeze(0).expand(bsz, -1)

        temp_control_user = self.wte_user(input_tokens_user)
        past_key_values_user = self.control_trans_user(temp_control_user)  # bsz, seqlen, layer*emb
        past_key_values = past_key_values_user

        bsz, seqlen, _ = past_key_values.shape
        past_key_values = past_key_values.view(bsz, seqlen, self.match_n_layer * 2, self.match_n_head,
                                               self.match_n_embd)
        past_key_values = self.dropout(past_key_values)
        past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)

        result = []
        for _, key_val in enumerate(past_key_values):
            result.append(
                {
                    "prev_key": key_val[0].contiguous(),
                    "prev_value": key_val[1].contiguous(),
                    "prev_key_padding_mask": torch.zeros(bsz, seqlen).to(key_val.device).bool()  # bsz, preseqlen
                }
            )
        return result

    def forward(
            self,
            input_ids,
            attention_mask,
            labels,
            x_attention_mask=None,
            label_ids=None,
            multi_labels=None,
    ):
        if isinstance(input_ids, list):
            bsz = input_ids[0].shape[0]
        else:
            bsz = input_ids.shape[0]

        if isinstance(x_attention_mask, list):
            x_attention_mask = x_attention_mask[0]

        past_prompt = self.get_prompt(bsz=bsz)

        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            labels=labels,
            output_hidden_states=True
        )
        hidden_states = output.hidden_states[-1]
        # 拿出 X 的 embedding, 顺便将其它 token 的 embedding 遮掩为 0
        x_attention_mask = x_attention_mask[:, :, None].expand(-1, -1, hidden_states.shape[-1])
        x_hidden_state = hidden_states * x_attention_mask
        # 输入到 transformer 层
        transformer_layer_out = self.transformer_layer(x_hidden_state)
        # 向量取平均值，输入到 2层 MLP 进行分类
        last_hidden_states = transformer_layer_out.mean(dim=1)
        cls_logits, features = self.classifier.forward(last_hidden_states)
        # 获取联合损失
        gen_loss = output.loss
        cls_loss = None
        ce_loss = None
        total_loss = 0
        if label_ids is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            ce_loss = loss_fct(cls_logits, label_ids.view(-1))
            loss_fct = MultiLabelCircleLoss()
            multi_loss = loss_fct.forward(cls_logits, multi_labels)
            cls_loss = ce_loss * self.m_loss + multi_loss * (1-self.m_loss)
            total_loss = gen_loss * self.lamda_loss + cls_loss * (1-self.lamda_loss)

        return {'loss': total_loss, 'feature': features, 'logits': cls_logits,
                'gen_loss': gen_loss, 'cls_loss': cls_loss, 'ce_loss': ce_loss}

    def evaluate_step(self, input_ids, attention_mask, x_attention_mask=None):
        # beam 3 搜索
        generate_tokens = self.generate(input_ids, attention_mask=attention_mask,
                                        num_beams=3, num_return_sequences=3,
                                        return_dict_in_generate=True, output_scores=True,
                                        min_length=input_ids.shape[1] + 2, max_length=input_ids.shape[1] + 8,
                                        early_stopping=True)
        result = {'decoder_tokens': generate_tokens.sequences[:, input_ids.shape[1]:]}
        # 输入到 forward 去拿 分类的 logits
        result.update(self.forward(input_ids, attention_mask, labels=None, x_attention_mask=x_attention_mask))
        return result

    def generate(self,
                 input_ids,
                 attention_mask,
                 **kwargs):

        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(
            bsz=bsz, sample_size=kwargs['num_beams'],
        )
        generated_ids = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_prompt=past_prompt,
            use_cache=True,
            **kwargs,
        )

        return generated_ids
