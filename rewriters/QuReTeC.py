from pytorch_transformers import BertForTokenClassification, BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
from .AbstractRewriter import AbstractRewriter
import re
import json
import os

class Ner(BertForTokenClassification):
    """
    BERT based classifier from 	Nikos Voskarides, Dan Li, Pengjie Ren, Evangelos Kanoulas, Maarten de Rijke:
    Query Resolution for Conversational Search with Limited Supervision. SIGIR 2020: 921-930.

    The class is taken from: https://github.com/nickvosk/sigir2020-query-resolution
    """

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None, attention_mask_label=None, device="cuda:0"):

        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32).to(device)

        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class QuReTeC(AbstractRewriter):
    """
    Implementation of QuReTeC, following the AbstractRewriter structure.
    Notice that this implementation is thought to be used at inference time, possibly in online scenarios. It is not optimized to combine multiple
    training queries at once in a single tensor or to operate in parallel.
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: is expected to contain two values, "model_dir" and "device".
        The first defines where to find the pretained QuReTeC model. The second indicates the device where to do the torch computation.
        """
        super().__init__(**kwargs)
        self.model_dir = kwargs["model_dir"]
        self.device = kwargs.get("device", "cpu")
        #importantly, we expect a "train_args.json" in the model's directory where is defined the maximum model's sequence length
        self.max_seq_length =  json.load(open(os.path.join(self.model_dir, "train_args.json")))['max_seq_length']

        # we intialize the model and the tokenizer
        self.model = Ner.from_pretrained(self.model_dir).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_dir, do_lower_case=True)

        # a list of the possible token labels, not relevant, relevant, class token, sep token, and a mapper to intergers. used later.
        self.label_list = ["O", "REL", "[CLS]", "[SEP]"]
        self.label_map = {i: label for i, label in enumerate(self.label_list, 1)}


    def _preprocess_query(self, query: str) -> str:
        """
        This function processes the query text to make it suitable for QuReTeC.

        :param query:  a string rpresenting the query
        :return: a version of the query preprocessed according to QuReTec needs (spaces before special characters and also after hyphens"
        """


        # Regular expression for special characters (excluding spaces and word characters)
        pattern_special_chars = r'([^\w\s])'
        # Add space before special characters
        result = re.sub(pattern_special_chars, r' \1', query)
        # Add space after hyphens only
        result = re.sub(r'-(?=\S)', r'- ', result)

        if result[-1]!="?":
            result += " ?"

        return result.lower()

    def _add_to_history(self, preprocessed_query, **kwargs):
        """
        This function adds the query to the history. QuReTeC uses a simple concatenation of strings as history.

        :param preprocessed_query: the (preprocessed) version of the query
        """
        if self.history is None:
            self.history = preprocessed_query
        else:
            self.history += " " + preprocessed_query

    def rewrite(self, query, **kwargs):

        #first, we preprocess the query
        processed_query = self._preprocess_query(query)

        if self.history is None:
            # if the story is empty, then we simply need to return the query itself as no rewrite is needed.
            self._add_to_history(processed_query)
            return query

        else:
            #in case we have some history, we preprocess it and combine it with the current query
            input_query = self.history + " [SEP] " + processed_query
            #now that we have our input_query, we can add the query to history
            self._add_to_history(processed_query)


            #for the model to work, we need a vector with the tokens and a vector describing which tokens are "valid"
            textlist = input_query.split(' ')
            tokens = []
            valid = []

            for word in textlist:
                # tokenize the word using the bert tokenizer and add it to the tokens
                token = self.tokenizer.tokenize(word)
                tokens += token
                # sometimes, a word is split in multiple tokens, we keep track of the first of such tokens, considering it "valid"
                valid += [1] + [0] * (len(token) - 1)

            # we make sure that tokens and valid are below the length of the
            tokens = tokens[0:(self.max_seq_length - 2)]
            valid = valid[0:(self.max_seq_length - 2)]

            # we add some special token needed and pad the vectors
            pad_length = max(0, self.max_seq_length - len(tokens) - 2)
            ntokens = ["[CLS]"] + tokens + ["[SEP]"] + [self.tokenizer.pad_token_id] * pad_length
            valid = [1] + valid + [1] + [0] * pad_length
            input_mask = [1] * (len(tokens) + 2) + [0] * pad_length
            segment_ids = [0] * len(ntokens)

            # we need to identify where the current index starts
            cur_turn_index = tokens.index('[SEP]')
            label_mask = [1] * cur_turn_index + [0] * (len(ntokens) - cur_turn_index)

            # finally. we convert the tokens into ids
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)

            #all the vectors are converted into tensors and transmitted to the device where the computation will occurr
            input_ids = torch.tensor([input_ids], dtype=torch.long).to(self.device)
            input_mask = torch.tensor([input_mask], dtype=torch.long).to(self.device)
            segment_ids = torch.tensor([segment_ids], dtype=torch.long).to(self.device)
            valid_ids = torch.tensor([valid], dtype=torch.long).to(self.device)
            label_mask = torch.tensor([label_mask], dtype=torch.long).to(self.device)

            # we compute the probabilities of each label
            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, valid_ids=valid_ids, attention_mask_label=label_mask, device=self.device)

            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2).detach().cpu().numpy()[0]

            # we need to merge the tokens that were split during the tokenization using the bert tokenizer.
            rel_tokens = []
            for i in range(1, cur_turn_index + 1):
                if valid[i] == 1:
                    rel_tokens.append(ntokens[i])
                else:
                    rel_tokens[-1] = (rel_tokens[-1] + ntokens[i]).replace("##", "")

            # now, we can decide which tokens to keep, based on their relevance
            out_tokens = []
            for e, t in enumerate(rel_tokens):
                # we check which tokens were labeled as relevant and store them
                if self.label_map.get(logits[e+1], 'O') == "REL" and t not in out_tokens:
                    out_tokens.append(t)

            # if at least one new relevant token is found, we add it to the query
            if len(out_tokens) == 0:
                return query
            else:
                return query + " "  + " ".join(out_tokens)