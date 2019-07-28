"""Implements a builder for creating vectorizers."""
from word_vectorizer.exceptions.not_known_technology_exception \
    import NotKnownTechnologyException
from word_vectorizer.models.fasttext_vectorizer import FastTextVectorizer
from word_vectorizer.models.glove_vectorizer import GloveVectorizer
from word_vectorizer.models.model_data import ModelData
from word_vectorizer.models.vectorizer import Vectorizer
from word_vectorizer.models.word2vec_vectorizer import Word2VecVectorizer


class VectorizerBuilder:
    """Class that builds the suitable vectorizer for each technology."""
    _DICT_TECH_TO_CLASS = {"word2vec": Word2VecVectorizer,
                           "glove": GloveVectorizer,
                           "fasttext": FastTextVectorizer}

    _TAG_FOR_GENSIM_MODELS = "gensim"

    @classmethod
    def build_vectorizer(cls, data_of_model: ModelData, path_to_model: str) \
            -> Vectorizer:
        """"Builds the vectorizer.

        Args:
            data_of_model (ModelData): data of the model to be built.
            path_to_model (str): path to the location of the model in disk.
        """

        type_of_model = data_of_model.tech

        try:
            class_to_build = cls._DICT_TECH_TO_CLASS[type_of_model]
        except KeyError:
            raise NotKnownTechnologyException
        return class_to_build(data_of_model, path_to_model)
