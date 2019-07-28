from unittest import TestCase
from unittest.mock import patch

from word_vectorizer.model_downloading.google_drive_model_downloader import \
    GoogleDriveModelDownloader


def get_path_of_class(class_):
    return ".".join([class_.__module__, class_.__name__])


class TestGoogleDriveModelDownloader(TestCase):
    MODEL_1_GOOGLE_DRIVE_URL = \
        "https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit"
    NAME_MODEL_1 = "model_1"
    MODEL_1_ID_IN_GOOGLE_DRIVE = "0B7XkCwpI5KDYNlNUTTlSS21pQmM"

    @patch(get_path_of_class(GoogleDriveModelDownloader) +
           '._download_file_from_google_drive')
    def test_download_from_url_no_google_drive_OK(self,
                                                  mock_download_from_google):
        path = GoogleDriveModelDownloader.download_from_url(
            self.MODEL_1_GOOGLE_DRIVE_URL, self.NAME_MODEL_1)
        mock_download_from_google.assert_called_once_with(
            self.MODEL_1_ID_IN_GOOGLE_DRIVE,
            path)
        self.assertTrue(path.endswith(self.NAME_MODEL_1))
