
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import torch
import torch.nn.functional as F
from .AbstractRewriter import AbstractRewriter


def top_p_filtering(logits, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits



class Cqr(AbstractRewriter):
    """
    Conversational Query Rewriter (CQR), as proposed in by Shi Yu, Jiahua Liu, Jingqin Yang, Chenyan Xiong, Paul N. Bennett, Jianfeng Gao, Zhiyuan
    Liu: Few-Shot Generative Conversational Query Rewriting. SIGIR 2020: 1933-1936.

    The implementation is based on https://github.com/thunlp/ConversationQueryRewriter.
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        self.model_dir: str = kwargs["model_dir"]
        self.device = kwargs["device"]

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_dir).to(self.device).eval()

        # use the minimum between the model max sequence length and the user specified max seq len (if it was defined)
        self.max_seq_len : int = min(kwargs.get("max_seq_len", 20), self.model.config.max_position_embeddings)

        self.temperature : float = kwargs.get("temperature", 0.0)
        self.top_p : float = kwargs.get("top_p", 0.9)

        self.special_tokens = ['<SEP>', '<PAD>', '<BOS>', '<EOS>']

    def _add_to_history(self, processed_query, **kwargs):
        """
        :param preprocessed_query: the (preprocessed) version of the query.
        """
        if self.history is None:
            self.history = []
        self.history.append(processed_query)


    def rewrite(self, query: str, **kwargs):

        processed_query = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(query))

        if self.history is None:
            self._add_to_history(processed_query)
            return query
        else:
            input_ids = []
            for toks in self.history:
                input_ids += toks + [self.tokenizer.sep_token_id]
            input_ids += processed_query
            input_ids.append(self.tokenizer.bos_token_id)

            input_length = len(input_ids)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device).unsqueeze(0)

            with torch.no_grad():
                for _ in range(self.max_seq_len):
                    inputs = {'input_ids': input_ids}

                    outputs = self.model(**inputs)
                    next_token_logits = outputs[0][:, -1, :] / (self.temperature if self.temperature > 0 else 1.)

                    filtered_logits = top_p_filtering(next_token_logits, top_p=self.top_p)
                    if self.temperature == 0:  # greedy sampling:
                        next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
                    else:
                        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

                    new_token = (next_token).detach().cpu().tolist()
                    if self.tokenizer.decode(new_token[0]).strip() == "<EOS>":
                        break
                    input_ids = torch.cat((input_ids, next_token), dim=1)

            pred_ids = (input_ids[0, input_length:]).detach().cpu().tolist()
            pred_text = self.tokenizer.decode(pred_ids, clean_up_tokenization_spaces=True)
            pred_text = self.remove_special_tokens(pred_text)

            self._add_to_history(processed_query)

        return pred_text


    def remove_special_tokens(self, text):
        # Remove special tokens from the output text in rare cases
        for token in self.special_tokens:
            text = text.replace(token, "")
        return text
