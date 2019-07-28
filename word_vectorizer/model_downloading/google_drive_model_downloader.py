from os.path import isfile

import requests

from word_vectorizer.model_downloading.model_downloader import ModelDownloader


class GoogleDriveModelDownloader(ModelDownloader):
    """Downloads a model from  a Google Drive url."""

    _URL_GOOGLE_DRIVE = "https://docs.google.com/uc?export=download"
    _CHUNK_SIZE = 32768

    @classmethod
    def download_from_url(cls, url: str, name_model: str) -> str:

        destination = super().download_from_url(url, name_model)

        if not isfile(destination):
            id_ = cls._take_id_from_google_drive_url(url)
            cls._download_file_from_google_drive(id_, destination)

        return destination

    @classmethod
    def _take_id_from_google_drive_url(cls, url):
        x = url.split('/d/')[-1]
        return x.split('/')[0]

    @classmethod
    def _download_file_from_google_drive(cls, id_: str, destination: str) -> \
            str:

        session = requests.Session()

        response = session.get(cls._URL_GOOGLE_DRIVE,
                               params={'id': id_}, stream=True)
        token = cls._get_confirm_token(response)

        if token:
            params = {'id': id_, 'confirm': token}
            response = session.get(cls._URL_GOOGLE_DRIVE,
                                   params=params, stream=True)

        cls._save_response_content(response, destination)

        return destination

    @classmethod
    def _get_confirm_token(cls, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    @classmethod
    def _save_response_content(cls, response, destination):

        with open(destination, "wb") as f:
            for chunk in response.iter_content(cls._CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
