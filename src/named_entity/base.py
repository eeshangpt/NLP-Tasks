from abc import ABC, abstractmethod


class BaseRecognizer(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, texts):
        pass