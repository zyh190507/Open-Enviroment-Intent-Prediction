import sys

sys.path.append("../")

import torch
from torch import nn
from transformers import AutoTokenizer
from BART_prefix_tuning.base import PushToHubFriendlyModel, BARTClassificationHead
from BART_prefix_tuning.modeling_bart import BartForConditionalGeneration


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
        self.pretrain_model = BartForConditionalGeneration.from_pretrained(
            args.model_name
        )
        self.config = self.pretrain_model.config
        self.match_n_layer = self.config.decoder_layers
        self.match_n_head = self.config.decoder_attention_heads
        self.n_embd = self.config.d_model
        assert self.n_embd % self.match_n_head == 0
        self.match_n_embd = self.n_embd // self.match_n_head  # huggingface BART's dim of kv need to be calculated

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

        # 分类头
        self.transfomer_layer = torch.nn.TransformerEncoderLayer(d_model=self.n_embd,
                                                                 nhead=self.match_n_head, batch_first=True)
        self.classifer = BARTClassificationHead(input_dim=self.n_embd, inner_dim=self.n_embd,
                                                num_classes=args.num_labels, pooler_dropout=0.1)

        self.lamda_loss = args.lamda_loss

        # 冻结 bart 参数
        for param in self.pretrain_model.parameters():
            param.requires_grad = False

        # 计算 prefix train 的 参数量
        total_mem = 0
        for param in self.parameters():
            if param.requires_grad:
                total_mem += param.numel()

        # 计算 bart 的总参数量
        bart_men = 0
        for param in self.pretrain_model.parameters():
            bart_men += param.numel()

        print("total mem: {} bart mem: {}".format(total_mem, bart_men))

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
                labels=None,
                label_ids=None
                ):
        # 为每批数据生成对应的 prompt
        bsz = input_ids.shape[0]

        past_prompt = self.get_prompt(
            bsz=bsz,
        )

        outputs = self.pretrain_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_prompt=past_prompt,
        )
        # BartEncoder 的 last_hidden_state 来做分类, 先经过 transformer layer 再取平均
        encoder_last_hidden_state = outputs.encoder_last_hidden_state
        transformer_layers = self.transfomer_layer(encoder_last_hidden_state).mean(dim=1)
        cls_logits, cls_features = self.classifer(transformer_layers)

        # 计算联合损失
        gen_loss = outputs.loss
        total_loss = 0
        cls_loss = 0
        if label_ids is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            cls_loss = loss_fct(cls_logits, label_ids.view(-1))
            total_loss = gen_loss * self.lamda_loss + cls_loss * (1-self.lamda_loss)

        return {'loss': total_loss, "feature": cls_features, 'logits': cls_logits, 'gen_loss': gen_loss,
                'cls_loss': cls_loss}

    def evaluate_step(self,
                      input_ids,
                      attention_mask,
                      labels=None,
                      ):
        # 测试时候 labels 是 It was 。直接作为 decoder_input_ids 输入到 生成模型中
        decoder_input_ids = labels
        generate_tokens = self.generate(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        decoder_input_ids=decoder_input_ids,
                                        early_stopping=True,
                                        num_return_sequences=3,
                                        num_beams=3, output_scores=True, max_length=decoder_input_ids.shape[1] + 8,
                                        min_length=decoder_input_ids.shape[1] + 2,
                                        return_dict_in_generate=True,
                                        )
        result = {'decoder_tokens': generate_tokens.sequences[:, decoder_input_ids.shape[1]:]}
        # 输入到 forward 中以获取 分类的 logits
        result.update(self.forward(input_ids=input_ids, attention_mask=attention_mask))
        return result

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
