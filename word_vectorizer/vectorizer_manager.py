"""Vectorizer Manager.

Implements the class that manages the operations on the vectorizer manager.
"""
import logging
import os
import shutil
from os import listdir
from os.path import isfile, join
from typing import List

from word_vectorizer.constants import Constants
from word_vectorizer.exceptions.not_existing_model_exception import \
    NotExistingModelException
from word_vectorizer.model_downloading.model_downloader_getter import \
    ModelDownloaderGetter
from word_vectorizer.models.model_data import ModelData
from word_vectorizer.models.model_data_loader import ModelDataLoader
from word_vectorizer.models.vectorizer_builder import VectorizerBuilder


class VectorizerManager(object):
    """Class that manages the vectorizers."""

    def __init__(self):
        self.load_data_of_models()
        self.loaded_models = {}

    def load_data_of_models(self):
        """Loads the information of the models from a json conf file."""
        self.models_parameters = ModelDataLoader.load_data()

    def list_downloaded_models(self) -> List[str]:
        """Gives a list of all models that are currently in disk.
        Returns:
            A list of strings with the names of the downloaded models.
        """
        path_of_models = Constants.DESTINATION_FOLDER
        try:
            list_ = [f for f in listdir(path_of_models) if
                     isfile(join(path_of_models, f))]
        except FileNotFoundError:
            list_ = []
        return list_

    def list_loaded_models(self) -> List[str]:
        """Gives a list of all models that are currently in memory.
        Returns:
            A list of strings with the names of the loaded models.
        """
        return list(self.loaded_models.keys())

    def _check_if_model_is_downloaded(self, name_model):
        return name_model in self.list_downloaded_models()

    def load_model(self, name_model: str) -> None:
        """Loads a model to memory.

        Args:
            name_model (str): name of the model that wants to be loaded.
        Raises:
            NotUnderstoodURLException: when there is no implementation to
                           download from a given url.
            NotKnownTechnologyException: when the technology given in the
                                         info of the asked model cannot be
                                         managed.
        """
        if name_model in self.list_loaded_models():
            return None
        data_of_model = self._select_asked_model_data(name_model)

        path = ModelDownloaderGetter.get_downloader(data_of_model.url). \
            download_from_url(data_of_model.url, data_of_model.name)

        model = VectorizerBuilder.build_vectorizer(data_of_model, path)

        self.loaded_models[name_model] = model

    def _select_asked_model_data(self, name_of_model: str) -> ModelData:
        try:
            model = list(
                filter(lambda x: x.name == name_of_model,
                       self.models_parameters))[0]
        except IndexError:
            raise NotExistingModelException

        return model

    def vectorize(self, name_model, *args, **kwargs):
        """Uses a model to get the asked vectors.

        Args:
            name_model (str): name of the model that wants to be used. If it
            is not loaded, it is done before computing the vector.
            *args:  not keyed arguments to be passed to the vectorizer.
             **kwargs: keyed arguments to be passed to the vectorizer.
        Returns:
            A numpy.ndarray vector representing the meaning of the input word.
        Example:
            >>> VectorizerManager().vectorize("a_model_name", "hello")

        """
        if name_model not in self.list_loaded_models():
            logging.info("Loading model {}".format(name_model))
            self.load_model(name_model)

        return self.loaded_models[name_model](*args, **kwargs)

    def clear_loaded_models(self):
        """Removes all models loaded in memory"""
        del self.loaded_models
        self.loaded_models = {}

    def remove_all_models_from_disk(self) -> None:
        """Removes all permanent files from disk."""
        if os.path.exists(Constants.DESTINATION_FOLDER):
            shutil.rmtree(Constants.DESTINATION_FOLDER)

    def remove_model(self, name_model) -> None:
        """Removes a model from memory.

        Args:
            name_model (str): name of the model that wants to be removed
                              from memory.
        """
        try:
            del self.loaded_models[name_model]
        except KeyError:
            raise NotExistingModelException

    def list_available_models(self) -> List[str]:
        """Gets a list of all model for which there is data available and
        thus can be loaded.
        Returns:
            A list with the name of the models.
        """
        return [x.name for x in self.models_parameters]

    def describe_model(self, name_model):
        """Gets the information of an available model.

        Args:
            name_model (str): name of the model.
        Returns:
             A dict with the data of the model. For example:
             {"name": "glove-spanish-300.gz",
              "tech": "glove",
              "dimensions": 300,
              "language": "spanish",
               "url": "http://url/to/model/glove-sbwc
               .i25.vec.gz",
              "binary": false,
              "description": "Glove in spanish"}
        """
        model_data = self._select_asked_model_data(name_model)
        return dict(model_data)


if __name__ == "__main__":
    model = "word2vec_SBWC_spanish_300.txt.bz2"
    logging.basicConfig(level=logging.DEBUG)
    vec = VectorizerManager()
    vec.load_model(model)
    print(vec.vectorize(model, "hola"))
