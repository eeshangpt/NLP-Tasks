from typing import List, Dict

import pandas as pd
from transformers import pipeline, Pipeline

from .base import BaseOracle
from .exception import ContextError


class SimpleOracle(BaseOracle):
    """
    A simple oracle that takes in a list of questions and a list of contexts
    and returns a list of answers for each question.

    Attributes
    ----------
    _model_name: str
        The name of the model to use for question answering.
    _model: pipeline
        The question answering pipeline.

    Methods
    -------
    _initialize_model()
        Initialize the model.
    _generate_answers(questions: List[str], contexts: List[str])
        Generate answers for the questions.
    _post_process_answers(answers: List[Dict], questions: List[str])
        Post process the answers.
    predict(questions: List[str], contexts: List[str])
        Predict the answers for the questions.
    """

    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initialize the oracle.
        Parameters
        ----------
        model_name: str
            The name of the model to use for question answering.
        """
        self._model_name = model_name
        self._initialize_model()

    def _initialize_model(self) -> Pipeline:
        """
        Initialize the model.

        Returns
        -------
        Pipeline
        """
        self._model = pipeline("question-answering", model=self._model_name)

    def _generate_answers(self, questions: List[str], contexts: List[str]) -> List[str]:
        """
        Generates the answers

        Parameters
        ----------
        questions : List[str]
            Questions that are asked.
        contexts : List[str]
            Context that is used for answering the questions

        Returns
        -------
        List[str]
            Answers.

        Raises
        ------
        ContextError
            When the context is ambiguous or absent
        """
        if len(contexts) == 1:
            contexts = contexts[0]
        elif len(contexts) > 1:
            contexts = "\n".join(contexts)
        else:
            raise ContextError("No context provided.")
        return self._model(question=questions, context=contexts)

    @staticmethod
    def _post_process_answers(
        answers: List[Dict], questions: List[str]
    ) -> pd.DataFrame:
        """
        Post-processing the answers.

        Parameters
        ----------
        answers: List[dict]
            List of answers with other information.
        questions: List[str]
            List of all questions.

        Returns
        -------
        pd.DataFrame
            A structured output containing answers, questions, and score.
        """
        final_answers = [
            {"question": ques, **ans} for ques, ans in zip(questions, answers)
        ]
        answers = pd.DataFrame(final_answers)[["score", "answer", "question"]]
        return answers

    def predict(self, questions: List[str], contexts: List[str]) -> pd.DataFrame:
        """
        Generates the answers to the questions.

        Parameters
        ----------
        questions: List[str]
            All the questions that are being asked.
        contexts: List[str]
            Contexts for the questions to be answered.

        Returns
        -------
        pd.DataFrame
            Generated answers.
        """
        answers = self._generate_answers(questions, contexts)
        answers = self._post_process_answers(answers, questions)
        return answers


class TFOracle(BaseOracle):

    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        """
        Initialize the oracle.
        Parameters
        ----------
        model_name: str
            The name of the model to use for question answering.
        """
        self._model_name = model_name
        self._initialize_tokenizer()
        self._initialize_model()

    def _initialize_tokenizer(self):
        pass

    def _initialize_model(self) -> Pipeline:
        """
        Initialize the model.

        Returns
        -------
        Pipeline
        """
        self._model = None

    def _generate_answers(self, questions: List[str], contexts: List[str]) -> List[str]:
        pass
    def predict(self, texts):
        self._generate_answers(None, None)
