from abc import ABC, abstractmethod

from word_vectorizer.models.model_data import ModelData


class Vectorizer(ABC):
    """Abastract class that manages the vectors of words."""

    def __init__(self, model_data: ModelData, path_to_model: str):
        self.model_data = model_data
        self.path_to_model = path_to_model
        self.model = self._load_model(path_to_model)

    @abstractmethod
    def vectorize(self, *args, **kwargs):
        pass

    @abstractmethod
    def _load_model(self, path_to_model: str):
        pass

    def __call__(self, *args, **kwargs):
        return self.vectorize(*args, **kwargs)
