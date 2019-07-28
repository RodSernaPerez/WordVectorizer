"""Class that downloads the models."""
import logging
import os
from abc import ABC

from word_vectorizer.constants import Constants


class ModelDownloader(ABC):
    """Downloads a model from the internet.

    The class is implemented as an abstract class from which all type of
    downloaders should inherit and override the method 'download_from_url'.

    Example:
        >>> ModelDownloader.download_from_url("www.url.to.model.com",
                                              "name_model.gz")
             'path/to/where/models/are/saved/name_model.gz'
    """
    def __init__(self):
        pass

    @classmethod
    def _get_path_to_folder(cls):
        return Constants.DESTINATION_FOLDER

    @classmethod
    def _create_folder_for_model(cls, folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

    @classmethod
    def download_from_url(cls, url: str, name_model: str) -> str:
        """Downloads a model from an url.

        Args:
            name_model (str): name with which the model will be saved.
            url (str): url from which the model must be downloaded.
        Returns:
            Path in which model has been saved.
        """
        destination = os.path.join(cls._get_path_to_folder(), name_model)

        cls._create_folder_for_model(cls._get_path_to_folder())
        logging.info("Downloading model {name_model} to path {destination}")
        return destination
