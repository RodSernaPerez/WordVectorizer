from unittest import TestCase
from unittest.mock import patch, MagicMock

from word_vectorizer.exceptions.not_understood_url_exception import \
    NotUnderstoodURLException
from word_vectorizer.model_downloading.gensim_model_downloader import \
    GensimModelDownloader
from word_vectorizer.model_downloading.google_drive_model_downloader \
    import GoogleDriveModelDownloader
from word_vectorizer.model_downloading.model_downloader_getter import \
    ModelDownloaderGetter
from word_vectorizer.model_downloading.normal_url_model_downloader import \
    NormalUrLModelDownloader


def get_path_of_class(class_):
    return ".".join([class_.__module__, class_.__name__])


class TestModelDownloaderGetter(TestCase):
    MODEL_1_GOOGLE_DRIVE_URL = \
        "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit"
    MODEL_2_NON_GOOGLE_DRIVE_URL = \
        "http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300-min5" \
        ".txt.bz2 "
    MODEL_3_GENSIM_URL = "gensim"

    NOT_VALID_URL = "dajsafhalsids"

    def setUp(self) -> None:
        self.mock_google_drive_downloader = \
            patch(get_path_of_class(GoogleDriveModelDownloader)).start()
        self.mock_normal_url_downloader = \
            patch(get_path_of_class(NormalUrLModelDownloader)).start()
        self.mock_gensim_model_downloader = \
            patch(get_path_of_class(GensimModelDownloader)).start()

    def tearDown(self):
        patch.stopall()

    def test_get_downloader_Google_Drive_OK(self):
        mocking_ = MagicMock(GoogleDriveModelDownloader)
        self.mock_google_drive_downloader.return_value = \
            mocking_
        downloader = ModelDownloaderGetter.get_downloader(
            self.MODEL_1_GOOGLE_DRIVE_URL)

        self.assertTrue(isinstance(downloader, GoogleDriveModelDownloader))

    def test_get_downloader_non_Google_Drive_OK(self):
        mocking_ = MagicMock(NormalUrLModelDownloader)
        self.mock_normal_url_downloader.return_value = \
            mocking_
        downloader = ModelDownloaderGetter.get_downloader(
            self.MODEL_2_NON_GOOGLE_DRIVE_URL)

        self.assertTrue(isinstance(downloader, NormalUrLModelDownloader))

    def test_get_downloader_gensim_OK(self):
        mocking_ = MagicMock(GensimModelDownloader)
        self.mock_gensim_model_downloader.return_value = \
            mocking_
        downloader = ModelDownloaderGetter.get_downloader(
            self.MODEL_3_GENSIM_URL)

        self.assertTrue(isinstance(downloader, GensimModelDownloader))

    def test_not_valid_url_FAIL(self):
        with self.assertRaises(NotUnderstoodURLException):
            ModelDownloaderGetter.get_downloader(self.NOT_VALID_URL)
