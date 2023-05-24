from transformers import BartForConditionalGeneration
from torch import nn
import torch


class BARTClassificationHead(nn.Module):
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
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.nn.ReLU()(hidden_states)
        features = self.dropout(hidden_states)
        hidden_states = self.out_proj(features)
        return hidden_states, features


class Model(nn.Module):

    def __init__(self, args):
        super(Model, self).__init__()
        self.base_model = BartForConditionalGeneration.from_pretrained(args.model_name)
        self.config = self.base_model.config
        self.n_embd = self.config.d_model
        # print(self.config)
        self.lamda_loss = args.lamda_loss
        # 分类头
        self.classifer = BARTClassificationHead(input_dim=self.n_embd, inner_dim=self.n_embd,
                                                num_classes=args.num_labels, pooler_dropout=0.1)

        self.transfomer_layer = torch.nn.TransformerEncoderLayer(d_model=self.n_embd,
                                                                 nhead=self.config.decoder_attention_heads,
                                                                 batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, label_ids=None):
        # 根据 label 生成 decoder_input_ids, 即是右移一位
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        encoder_last_hidden_state = outputs.encoder_last_hidden_state.mean(dim=1)
        cls_logits, cls_features = self.classifer(encoder_last_hidden_state)
        # 计算联合损失
        gen_loss = outputs.loss
        total_loss = 0
        cls_loss = 0
        if label_ids is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            cls_loss = loss_fct(cls_logits, label_ids.view(-1))
            total_loss = gen_loss * self.lamda_loss + cls_loss * (1-self.lamda_loss)
        return {'loss': total_loss, "feature": cls_features, 'logits': cls_logits,
                'gen_loss': gen_loss, 'cls_loss': cls_loss, 'sent_emb': encoder_last_hidden_state}

    def evaluate_step(self,
                      input_ids,
                      attention_mask,
                      labels=None
                      ):
        # 生成 beam=3 的句子
        generate_tokens = self.base_model.generate(input_ids=input_ids,
                                                   attention_mask=attention_mask,
                                                   decoder_input_ids=labels,
                                                   early_stopping=True,
                                                   num_return_sequences=3,
                                                   num_beams=3, output_scores=True, max_length=8,
                                                   min_length=2,
                                                   return_dict_in_generate=True,
                                                   )
        result_dict = {}
        result_dict.update(self.forward(input_ids=input_ids, attention_mask=attention_mask, labels=labels))
        result_dict.update({'decoder_tokens': generate_tokens.sequences[:, labels.shape[1]:]})

        return result_dict
