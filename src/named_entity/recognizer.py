from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import (
    pipeline,
    TFBertForTokenClassification,
    BertTokenizer,
    BatchEncoding,
)

from .base import BaseRecognizer


class NamedEntityRecognizer(BaseRecognizer):
    """"""

    def __init__(
        self,
        model_name: str = "dslim/bert-base-NER",
        aggregation_strategy: str = "none",
    ):
        """"""
        self._model_name = model_name
        self._score_df = None
        self._aggregation_strategy = aggregation_strategy
        self._initialize_model()

    def _initialize_model(self):
        """"""
        self._model = pipeline("ner", model=self._model_name)

    @staticmethod
    def _post_process_entities(temp_df) -> List[List[str]]:
        """"""
        return temp_df[["entity_group", "word"]].to_numpy().tolist()

    def _predict_entities(self, texts: List[str]):
        """"""
        entities = self._model(texts, aggregation_strategy=self._aggregation_strategy)
        self._score_df = pd.DataFrame()
        for itr, ent_predicts in enumerate(entities):
            temp_ = pd.DataFrame(ent_predicts)
            temp_["sentence_num"] = itr + 1
            self._score_df = pd.concat([self._score_df, temp_], ignore_index=True)
        return (
            self._score_df.groupby(by=["sentence_num"])
            .apply(self._post_process_entities)
            .to_numpy()
            .tolist()
        )

    def predict(self, texts: List[str]):
        """"""
        return self._predict_entities(texts)


class TFNamedEntityRecognizer(BaseRecognizer):
    """"""

    def __init__(
        self,
        model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english",
        tokenizer_model_name: str = None,
    ):
        """"""
        self._model_name = model_name
        if tokenizer_model_name is None:
            tokenizer_model_name = model_name
        self._tokenizer_model_name = tokenizer_model_name
        self._initialize_tokenizer()
        self._initialize_model()

    def _initialize_tokenizer(self):
        """"""
        self._tokenizer = BertTokenizer.from_pretrained(self._model_name)

    def _initialize_model(self):
        """"""
        self._model = TFBertForTokenClassification.from_pretrained(
            self._model_name, from_pt=True
        )

    def _tokenize(self, texts) -> Tuple[List, BatchEncoding]:
        """"""
        tokens = self._tokenizer.encode_plus(texts, return_tensors="tf")
        predicted_tokens = self._tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
        return predicted_tokens, tokens

    def _predict_tokens(self, tokens) -> np.ndarray:
        """"""
        output = self._model(tokens)
        predicted_labels = tf.argmax(output.logits, axis=2).numpy()
        return predicted_labels

    def predict(self, texts: str) -> List:
        """"""
        predicted_tokens, tokens = self._tokenize(texts)
        predicted_labels = self._predict_tokens(tokens)
        return [
            (tkn, self._model.config.id2label[lbl])
            for tkn, lbl in zip(predicted_tokens, predicted_labels[0])
        ]
