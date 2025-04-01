from pytorch_transformers import BertForTokenClassification, BertTokenizer

import torch
from torch import nn
import torch.nn.functional as F
from .AbstractRewriter import AbstractRewriter
import re

from transformers import T5ForConditionalGeneration, T5Tokenizer
from spacy.lang.en import English
from typing import Optional


class Ntr(AbstractRewriter):
    """
     Neural Transfer Reformulation (NTR) as described by Sheng-Chieh Lin, Jheng-Hong Yang, Rodrigo Nogueira, Ming-Feng Tsai, Chuan-Ju Wang,
     Jimmy Lin: Multi-Stage Conversational Passage Retrieval: An Approach to Fusing Term Importance Estimation and Neural Query Rewriting. ACM Trans.
     Inf. Syst. 39(4): 48:1-48:29 (2021)


     Based on the implementation available on github: https://github.com/castorini/chatty-goose/tree/c7d0cd8c45354b09b5fb930ab0b5af8be2e5772b
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: is expected to contain two values, "model_name" and "max_length", "num_beans", "early_stopping".
        The parameters define where to get the model (if not specified, the huggingface model will be downloaded), the max length of the context
        (by default 64), the number of generation beams, and whether to use early stopping or not.
        """
        super().__init__(**kwargs)
        self.device = kwargs["device"]
        self.model_name : str = kwargs.get("model_name", "castorini/t5-base-canard")
        self.max_length : int = kwargs.get("max_length", 64)
        self.num_beams : int = kwargs.get("num_beams", 10)
        self.early_stopping : bool = kwargs.get("early_stopping", True)


        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)\
                                               .to(self.device)\
                                               .eval()

        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.nlp = English()

    def _add_to_history(self, context, **kwargs):
        """
        This function adds the query to the history. NTR uses both previous queries and answers, joined with a special symbol (|||).

        :param preprocessed_query: the (preprocessed) version of the query
        """
        if self.history is None:
            self.history = {"answer": [], "query": []}

        if "query" in context:
            self.history["query"].append(context["query"])
        if "answer" in context:
            self.history["answer"].append(context["answer"])

    def rewrite(self, query: str, previous_answer: Optional[str] = None, **kwargs):


        if self.history is None:
            # if the story is empty, then we simply need to return the query itself as no rewrite is needed.
            self._add_to_history({"query": query})
            return query

        else:
            if previous_answer:
                self._add_to_history({"answer": previous_answer})

            #in case we have some history, we preprocess it and combine it with the current query.
            #Ntr allows combining the current query with all the previous queries and the last answer
            input_query = " ||| ".join(self.history["query"] + self.history["answer"][-1:] + [query])

            #the current query is added to the history, before rewriting
            self._add_to_history({"query": query})


            src_text = " ".join([tok.text for tok in self.nlp(input_query)])

            input_ids = self.tokenizer(src_text, return_tensors="pt", add_special_tokens=True).input_ids.to(self.device)

            # Generate new sequence
            output_ids = self.model.generate(input_ids, max_length=self.max_length, num_beams=self.num_beams, early_stopping=self.early_stopping)

            # Decode output
            rewrite_text = self.tokenizer.decode(output_ids[0, 0:], clean_up_tokenization_spaces=True, skip_special_tokens=True)

            return rewrite_text


