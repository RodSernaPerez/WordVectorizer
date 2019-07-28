"""Model Downloading Management.

Classes to manage the automatic download of embedding models from the
internet."""

from word_vectorizer.model_downloading.model_downloader import ModelDownloader
from word_vectorizer.model_downloading.model_downloader_getter import \
    ModelDownloaderGetter

_all__ = [ModelDownloaderGetter.__name__, ModelDownloader.__name__]
