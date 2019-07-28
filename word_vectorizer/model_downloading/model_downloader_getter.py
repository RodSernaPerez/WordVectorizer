"""Class ModelDownloaderGetter

Implements a factory for getting model downloaders."""
from urllib.parse import urlparse

from word_vectorizer.exceptions.not_understood_url_exception import \
    NotUnderstoodURLException
from word_vectorizer.model_downloading.gensim_model_downloader import \
    GensimModelDownloader
from word_vectorizer.model_downloading.google_drive_model_downloader import \
    GoogleDriveModelDownloader
from word_vectorizer.model_downloading.model_downloader import ModelDownloader
from word_vectorizer.model_downloading.normal_url_model_downloader import \
    NormalUrLModelDownloader


class ModelDownloaderGetter:
    """Gets a suitable downloader.

    The class acts a factory for building downloaders according to the url
    from which model will be downloaded.

    Example:
        >>> downloader = ModelDownloaderGetter.get_downloader(
                                                  "www.url.to.model.com")

    """

    @classmethod
    def get_downloader(cls, url: str) -> ModelDownloader:
        """Get downloader.
        Gets a suitable downloader for getting the model from an url.

        Args:
            url (str): url to the repository from where the model can be
                       downloaded.
        Returns:
            An instance of ModelDownloader or any of its subclasses.
        Raises:
            NotUnderstoodURLException: when there is no implementation to
                                       download from a given url.
        """
        if cls._check_if_is_url(url):
            if cls._check_is_google_drive_url(url):
                x = GoogleDriveModelDownloader
            else:
                x = NormalUrLModelDownloader
        else:
            if url == "gensim":
                x = GensimModelDownloader
            else:
                raise NotUnderstoodURLException()
        return x()

    @classmethod
    def _check_is_google_drive_url(cls, url: str):
        return ".google.com" in url

    @classmethod
    def _check_if_is_url(cls, url):
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def __init__(self, ):
        pass
