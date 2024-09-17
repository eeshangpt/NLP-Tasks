from typing import List

import pandas as pd
from transformers import pipeline

from .base import BaseClassifier


class ZeroShotClassifier(BaseClassifier):
    """
    A class for zero-shot classification.

    Attributes
    ----------
    _model_name: str
        The name of the model to use for classification.
    _categories: List
        The list of categories to classify.
    _score_df: pd.DataFrame
        The dataframe containing the scores for each category.
    _threshold: float
        The threshold for selecting categories.
    _is_multi_label: bool
        Whether the model is a multi-label classifier.

    Methods
    -------
    _predict_categories(text)
        Predict a single category of the text.
    _predict_multiple_categories(text)
        Predict multiple categories of the text.
    _calculate_final_classes()
        Calculate the final predicted classes.

    predict(text)
        Predict the categories of the text.
    """

    def __init__(
        self,
        categories: List,
        model_name: str = "facebook/bart-large-mnli",
        is_multi_label: bool = False,
        threshold: float = 0.8,
    ):
        """
        Parameters
        ----------
        categories: List
            The list of categories to classify.
        model_name: str
            The name of the model to use for classification.
        is_multi_label: bool
            Whether the model is a multi-label classifier.
        threshold: float
            The threshold for selecting categories.
        """
        self._model_name = model_name
        self._categories = categories
        self._threshold = threshold
        self._is_multi_label = is_multi_label
        self._score_df = None
        self._initialize_model()

    def _initialize_model(self):
        """
        Initialize the model.

        Returns
        -------
        None
        """
        self.model = pipeline("zero-shot-classification", model=self._model_name)

    def _predict_categories(self, text) -> None:
        """
        Predict a single category of the text.
        Parameters
        ----------
        text: str
            The text to predict.

        Returns
        -------
        None
        """
        # Use the classifier to get the scores for each category
        output = self.model(
            text, candidate_labels=self._categories, multi_label=self._is_multi_label
        )

        # Create a dataframe from the scores
        self._score_df = pd.DataFrame(
            {"category": output["labels"], "score": output["scores"]}
        )

    def _predict_multiple_categories(self, text: str) -> None:
        """
        Predict multiple categories of the text.
        Parameters
        ----------
        text: str
            The text to predict.

        Returns
        -------
        None
        """
        self._predict_categories(text)

        self._score_df["selected_label"] = self._score_df["score"].apply(
            lambda x: x >= self._threshold
        )

    def _calculate_final_classes(self):
        """
        Calculate the final predicted classes.

        Returns
        -------
        List
            The final predicted classes.
        """
        if self._is_multi_label:
            return self._score_df[self._score_df.selected_label][
                "category"
            ].values.tolist()
        else:
            return self._score_df[self._score_df.score == self._score_df.score.max()][
                "category"
            ].values.tolist()

    def predict(self, text: str) -> List:
        """
        Predict the categories of the text.
        Parameters
        ----------
        text: str
            The text to predict.

        Returns
        -------
        List
            The predicted categories.
        """
        if self._is_multi_label:
            self._predict_multiple_categories(text)
        else:
            self._predict_categories(text)

        return self._calculate_final_classes()
