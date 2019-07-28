from unittest import TestCase
from unittest.mock import patch

from word_vectorizer.constants import Constants
from word_vectorizer.model_downloading.gensim_model_downloader import \
    GensimModelDownloader


class TestGensimModelDownloader(TestCase):
    NAME_MODEL = "name_model"
    URL = "gensim"

    PATH_WHERE_GENSIM_DOWNLOADS_MODEL = "this/is/a/path/to/the/" + NAME_MODEL
    PATH_TO_FOLDER_WHERE_GENSIM_DOWNLOADS = "this/is/a/path"

    @patch(GensimModelDownloader.__module__ + ".shutil", spec=True)
    @patch(GensimModelDownloader.__module__ + ".api")
    def test_download_from_url(self, mock_api, mock_shutil):
        mock_api.load.return_value = self.PATH_WHERE_GENSIM_DOWNLOADS_MODEL
        path = GensimModelDownloader.download_from_url(self.URL,
                                                       self.NAME_MODEL)
        mock_shutil.move.assert_called_once_with(
            self.PATH_WHERE_GENSIM_DOWNLOADS_MODEL,
            Constants.DESTINATION_FOLDER + "/" + self.NAME_MODEL)
        mock_shutil.rmtree.assert_called_once_with(
            self.PATH_TO_FOLDER_WHERE_GENSIM_DOWNLOADS)
        self.assertTrue(path.endswith(self.NAME_MODEL))
