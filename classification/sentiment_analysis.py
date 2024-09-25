from typing import List

import pandas as pd
import tensorflow as tf
from transformers import (
    DistilBertTokenizer,
    TFAutoModelForSequenceClassification,
    pipeline,
)

from .base import BaseClassifier


class SentimentClassifier(BaseClassifier):
    """
    A class for sentiment analysis.

    Attributes
    ----------
    _model_name: str
        The name of the model to use for classification.
    _score_df: pd.DataFrame
        The dataframe containing the scores for each category.
    model: pipeline
        The sentiment analysis pipeline.

    Methods
    -------
    _predict_sentiment(text)
        Predict the sentiment of the text.
    _calculate_final_sentiment()
        Calculate the final predicted sentiment.
    predict(text)
        Predict the sentiment of the text.
    """

    def __init__(
        self,
        model_name: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    ):
        """
        Initialize the sentiment analysis classifier.

        Parameters
        ----------
        model_name: str
            The name of the model to use for classification.
        """
        self._model_name = model_name
        self._score_df = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the model.

        Returns
        -------
        None
        """
        self.model = pipeline("sentiment-analysis", model=self._model_name)

    def _predict_sentiment(self, text: List[str]):
        """
        Predict the sentiment of the text.

        Parameters
        ----------
        text: List[str]
            The texts to predict the sentiment of.

        Returns
        -------
        None
        """
        # Use the classifier to get the scores for each category
        output = self.model(text)
        output = {k: [d[k] for d in output] for k in output[0]}

        self._score_df = pd.DataFrame(
            {"sentiment": output["label"], "score": output["score"]}
        )

    def _calculate_final_sentiment(self):
        """
        Calculate the final predicted sentiment.

        Returns
        -------
        List
            The final predicted sentiments.
        """
        return self._score_df["sentiment"].values.tolist()

    def predict(self, text: List[str]):
        """
        Predict the sentiment of the text.

        Parameters
        ----------
        text: List[str]
            The texts to predict the sentiment of.

        Returns
        -------
        List
            The final predicted sentiments.
        """
        self._predict_sentiment(text)
        return self._calculate_final_sentiment()


class TFSentimentClassifier(BaseClassifier):
    """
    A class for sentiment analysis using TensorFlow.

    Attributes
    ----------
    model_name: str
        The name of the model to use for classification.
    tokenizer_model: str
        The name of the tokenizer model to use.
    model: TFAutoModelForSequenceClassification
        The sentiment analysis model.

    Methods
    -------
    _initialize_tokenizer()
        Initialize the tokenizer.
    _initialize_model()
        Initialize the model.
    predict(text)
        Predict the sentiment of the text.
    """
    def __init__(
        self,
        model_name: str = "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer_model: str = None,
    ):
        """
        Initialize the sentiment analysis classifier.
        Parameters
        ----------
        model_name: str
            The name of the model to use for classification
        tokenizer_model: str
            The name of the tokenizer model to use. If None, the model name is used.
        """
        self.model_name = model_name
        if tokenizer_model is None:
            tokenizer_model = model_name
        self.tokenizer_model = tokenizer_model
        self._initialize_tokenizer()
        self._initialize_model()

    def _initialize_tokenizer(self):
        """
        Initialize the tokenizer.

        Returns
        -------
        None
        """
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)

    def _initialize_model(self):
        """
        Initialize the model.

        Returns
        -------
        None
        """
        self.model = TFAutoModelForSequenceClassification.from_pretrained(self.tokenizer_model)

    def predict(self, text: str) -> str:
        """
        Predict the sentiment of the text.

        Parameters
        ----------
        text: str
            The text to predict the sentiment of.

        Returns
        -------
        str
            The predicted sentiment.
        """
        input_tokens = self.tokenizer(text, return_tensors="tf")
        output_logit = self.model(**input_tokens).logits
        predicted_class_id = int(tf.math.argmax(output_logit, axis=-1)[0])
        return self.model.config.id2label[predicted_class_id]


class TextClassifier(BaseClassifier):
    """
    A class for text classification in multiple categories.

    Attributes
    ----------
    _model_name: str
        The name of the model to use for classification.
    _score_df: pd.DataFrame
        The dataframe containing the scores for each category.
    model: pipeline
        The sentiment analysis pipeline.

    Methods
    -------
    _initialize_model()
        Initialize the model.
    _predict_sentiment(text)
        Predict the sentiment of the text.
    _calculate_final_sentiment()
        Calculate the final predicted sentiment.
    predict(text)
        Predict the sentiment of the text.
    """
    def __init__(
        self,
        model_name: str = "SamLowe/roberta-base-go_emotions",
    ):
        """
        Initialize the sentiment analysis classifier.

        Parameters
        ----------
        model_name: str
            The name of the model to use for classification
        """
        self._model_name = model_name
        self._score_df = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the model.

        Returns
        -------
        None
        """
        self.model = pipeline("text-classification", model=self._model_name)

    def _predict_sentiment(self, text: List[str]):
        """
        Predict the sentiment of the text.

        Parameters
        ----------
        text: List[str]
            The texts to predict the sentiment of.

        Returns
        -------
        None
        """
        output = self.model(text)
        output = {k: [d[k] for d in output] for k in output[0]}

        self._score_df = pd.DataFrame(
            {"sentiment": output["label"], "score": output["score"]}
        )

    def _calculate_final_sentiment(self):
        """
        Calculate the final predicted sentiment.

        Returns
        -------
        List
            The final predicted sentiments.
        """
        return self._score_df["sentiment"].values.tolist()

    def predict(self, text: List[str]):
        """
        Predict the sentiment of the text.

        Parameters
        ----------
        text:   List[str]
            The texts to predict the sentiment of.

        Returns
        -------
        List
            The final predicted sentiments.
        """
        self._predict_sentiment(text)
        return self._calculate_final_sentiment()


class FinanceSentimentClassifier(BaseClassifier):
    """
    """
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
    ):
        """
        Initialize the sentiment analysis classifier.

        Parameters
        ----------
        model_name: str
            The name of the model to use for classification
        """
        self._model_name = model_name
        self._score_df = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the model.

        Returns
        -------
        None
        """
        self.model = pipeline("text-classification", model=self._model_name)

    def _predict_sentiment(self, text: List[str]):
        """
        Predict the sentiment of the text.

        Parameters
        ----------
        text: List[str]
            The texts to predict the sentiment of.

        Returns
        -------
        None
        """
        output = self.model(text)
        output = {k: [d[k] for d in output] for k in output[0]}

        self._score_df = pd.DataFrame(
            {"sentiment": output["label"], "score": output["score"]}
        )

    def _calculate_final_sentiment(self):
        """
        Calculate the final predicted sentiment.

        Returns
        -------
        List
            The final predicted sentiments.
        """
        return self._score_df["sentiment"].values.tolist()

    def predict(self, text: List[str]):
        """
        Predict the sentiment of the text.

        Parameters
        ----------
        text:   List[str]
            The texts to predict the sentiment of.

        Returns
        -------
        List
            The final predicted sentiments.
        """
        self._predict_sentiment(text)
        return self._calculate_final_sentiment()