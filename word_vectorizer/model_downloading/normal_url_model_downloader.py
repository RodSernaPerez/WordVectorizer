"""
Implementation for downloading from a normal url.
"""
import logging
import os
import urllib
from os.path import isfile

from word_vectorizer.model_downloading.model_downloader import ModelDownloader
from word_vectorizer.utils.progess_bar import DownloadProgressBar


class NormalUrLModelDownloader(ModelDownloader):
    """NormalUrlModelDownloader class

    Downloads a model from a normal url (not google drive, dropbox ...).
    """

    @classmethod
    def download_from_url(cls, url: str, name_model: str) -> str:
        destination = os.path.join(cls._get_path_to_folder(), name_model)
        if not isfile(destination):
            super().download_from_url(url, name_model)
            urllib.request.urlretrieve(url, destination, DownloadProgressBar())
        logging.debug("Save to destination: {}".format(destination))
        return destination
