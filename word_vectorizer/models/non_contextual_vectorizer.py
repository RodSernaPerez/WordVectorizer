from abc import ABC, abstractmethod

from word_vectorizer.exceptions.vector_not_computed_exception import \
    VectorNotComputedException
from word_vectorizer.models.vectorizer import Vectorizer


class NonContextualVectorizer(Vectorizer, ABC):
    """Class to manage a non contextual vectorizer."""

    def vectorize(self, word):
        try:
            vector = self.model[word]
        except KeyError:
            raise VectorNotComputedException
        return vector

    @abstractmethod
    def _load_model(self, path_to_model: str):
        pass

    def __call__(self, *args, **kwargs):
        return self.vectorize(*args, **kwargs)
