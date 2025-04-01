from typing import List

import torch
from torch import nn
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
import torch.nn.functional as F


from .AbstractBiencoder import AbstractBiencoder


class ConvDR(AbstractBiencoder):

    def __init__(self, **kwargs):

        self.model_path = kwargs["model_path"]
        self.device = kwargs.get('device', torch.device("cuda" if torch.cuda.is_available() else "cpu"))


        self.config = RobertaConfig.from_pretrained(self.model_path, num_labels=2, finetuning_task="MSMarco")
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_path, do_lower_case=True)
        self.model = RobertaDot_NLL_LN.from_pretrained(self.model_path, from_tf=bool(".ckpt" in self.model_path), config=self.config).to(self.device)
        self.max_query_len = 64
        self.max_docs_len = 512
        self.history = None

    def reset_history(self):
        self.history = None

    def _add_to_history(self, processed_query):
        if self.history is None:
            self.history = processed_query + [self.tokenizer.sep_token]
        else:
            self.history +=  [self.tokenizer.cls_token] + processed_query + [self.tokenizer.sep_token]



    def encode_query(self, query: str):

        tokens = self.tokenizer.tokenize(query)
        self._add_to_history(tokens)

        context_tokens = [self.tokenizer.cls_token] + self.history[-(self.max_query_len-1):]

        enc_tokens = self.tokenizer.encode(context_tokens+ [self.tokenizer.pad_token_id] * (self.max_query_len - len(context_tokens)),
                                           add_special_tokens=True,
                                           max_length=self.max_query_len,
                                           truncation=True)

        input_ids = [enc_tokens]
        att_masks = [[1] * len(context_tokens) + [0] * (self.max_query_len - len(context_tokens))]

        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long).to(self.device),
            "attention_mask": torch.tensor(att_masks, dtype=torch.long).to(self.device),
        }


        with torch.no_grad():
            embs = self.model.query_emb(**inputs).detach().cpu().numpy()
        return embs

    def encode_documents(self, texts: List[str]):
        input_ids = []
        att_masks = []
        for t in texts:
            tokens =  [self.tokenizer.cls_token] + self.tokenizer.tokenize(t)[:self.max_docs_len-2] + [self.tokenizer.sep_token]
            enc_tokens = self.tokenizer.encode(tokens + [self.tokenizer.pad_token_id] * (self.max_docs_len - len(tokens)),
                                               add_special_tokens=True,
                                               max_length=self.max_docs_len,
                                               truncation=True)

            input_ids.append(enc_tokens)
            att_masks.append([1] * len(tokens) + [0] * (self.max_docs_len - len(tokens)))

        inputs = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long).to(self.device),
            "attention_mask": torch.tensor(att_masks, dtype=torch.long).to(self.device),
        }
        with torch.no_grad():
            embs = self.model.body_emb(**inputs).detach().cpu().numpy()
        return embs


class EmbeddingMixin:
    """
    Mixin for common functions in most embedding models. Each model should define its own bert-like backbone and forward.
    We inherit from RobertaModel to use from_pretrained
    """
    def __init__(self, model_argobj):
        if model_argobj is None:
            self.use_mean = False
        else:
            self.use_mean = model_argobj.use_mean
        print("Using mean:", self.use_mean)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)

    def masked_mean(self, t, mask):
        s = torch.sum(t * mask.unsqueeze(-1).float(), axis=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d

    def masked_mean_or_first(self, emb_all, mask):
        # emb_all is a tuple from bert - sequence output, pooler
        if self.use_mean:
            return self.masked_mean(emb_all.last_hidden_state[0], mask)
        else:
            return emb_all[0][:, 0]

    def query_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")

    def body_emb(self, input_ids, attention_mask):
        raise NotImplementedError("Please Implement this method")


class NLL(EmbeddingMixin):
    def forward(self,
                query_ids,
                attention_mask_q,
                input_ids_a=None,
                attention_mask_a=None,
                input_ids_b=None,
                attention_mask_b=None,
                is_query=True):
        if input_ids_b is None and is_query:
            return self.query_emb(query_ids, attention_mask_q)
        elif input_ids_b is None:
            return self.body_emb(query_ids, attention_mask_q)

        q_embs = self.query_emb(query_ids, attention_mask_q)
        a_embs = self.body_emb(input_ids_a, attention_mask_a)
        b_embs = self.body_emb(input_ids_b, attention_mask_b)

        logit_matrix = torch.cat([(q_embs * a_embs).sum(-1).unsqueeze(1),
                                  (q_embs * b_embs).sum(-1).unsqueeze(1)],
                                 dim=1)  # [B, 2]
        lsm = F.log_softmax(logit_matrix, dim=1)
        loss = -1.0 * lsm[:, 0]
        return (loss.mean(), )


class RobertaDot_NLL_LN(NLL, RobertaForSequenceClassification):
    """None
    Compress embedding to 200d, then computes NLL loss.
    """
    def __init__(self, config, model_argobj=None):
        NLL.__init__(self, model_argobj)
        RobertaForSequenceClassification.__init__(self, config)
        self.embeddingHead = nn.Linear(config.hidden_size, 768)
        self.norm = nn.LayerNorm(768)
        self.apply(self._init_weights)

    def query_emb(self, input_ids, attention_mask):
        outputs1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        full_emb = self.masked_mean_or_first(outputs1, attention_mask)
        query1 = self.norm(self.embeddingHead(full_emb))
        return query1

    def body_emb(self, input_ids, attention_mask):
        return self.query_emb(input_ids, attention_mask)


