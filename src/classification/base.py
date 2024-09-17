from abc import ABC, abstractmethod


class BaseClassifier(ABC):
    @abstractmethod
    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict(self, text):
        pass
