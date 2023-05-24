import sys

import torch
from torch import nn

from transformers import GPT2LMHeadModel


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

        self.lamda_loss = args.lamda_loss

        # 分类头
        self.transformer_layer = torch.nn.TransformerEncoderLayer(d_model=self.base_model.config.n_embd,
                                                                  nhead=self.config.n_head, batch_first=True)
        self.classifier = GPT2ClassificationHead(self.base_model.config.n_embd, self.base_model.config.n_embd,
                                                 args.num_labels, self.base_model.config.resid_pdrop)

        ###### NUM PARAMS #########
        total_param = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)
                total_param += param.numel()
        print('total param is {}'.format(total_param))

    def forward(
            self,
            input_ids,
            attention_mask,
            labels,
            x_attention_mask=None,
            label_ids=None,
    ):
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
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
        total_loss = 0
        if label_ids is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            cls_loss = loss_fct(cls_logits, label_ids.view(-1))
            total_loss = gen_loss * self.lamda_loss + cls_loss

        return {'loss': total_loss, 'feature': features, 'logits': cls_logits,
                'gen_loss': gen_loss, 'cls_loss': cls_loss, 'sent_emb': x_hidden_state.mean(dim=1)}

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

        generated_ids = self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            **kwargs,
        )

        return generated_ids
