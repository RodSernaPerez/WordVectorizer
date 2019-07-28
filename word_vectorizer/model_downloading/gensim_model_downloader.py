"""Manages the donwload of a gensim default model."""
import shutil
from os.path import isfile

import gensim.downloader as api

from word_vectorizer.model_downloading.model_downloader import ModelDownloader


class GensimModelDownloader(ModelDownloader):
    """Manages the download of a model included in gensim."""

    @classmethod
    def download_from_url(cls, url: str, name_model: str) -> str:
        destination = super().download_from_url(url, name_model)
        if not isfile(destination):
            path_of_file = api.load(name_model.split(".")[0], return_path=True)
            cls._move_file_to_folder(path_of_file, destination)
        return destination

    @classmethod
    def _move_file_to_folder(cls, current_path, destination):
        shutil.move(current_path, destination)
        shutil.rmtree("/".join(current_path.split("/")[0:-3]))
