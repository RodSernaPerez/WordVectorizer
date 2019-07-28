from unittest import TestCase
from unittest.mock import patch, MagicMock

from word_vectorizer.model_downloading.normal_url_model_downloader import \
    NormalUrLModelDownloader
from word_vectorizer.utils.progess_bar import DownloadProgressBar


class TestNormalUrLModelDownloader(TestCase):
    NORMAL_URL = "http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors" \
                 "-300-min5.txt.bz2 "
    MODEL_NAME = "model"

    @patch(NormalUrLModelDownloader.__module__ + "." +
           DownloadProgressBar.__name__)
    @patch('urllib.request.urlretrieve')
    def test_download_from_url(self, moch_urlretrieve, mock_progress_bar):
        progress_bar = MagicMock(DownloadProgressBar)
        mock_progress_bar.return_value = progress_bar
        path = NormalUrLModelDownloader.download_from_url(self.NORMAL_URL,
                                                          self.MODEL_NAME)
        moch_urlretrieve.assert_called_once_with(self.NORMAL_URL, path,
                                                 progress_bar)

        self.assertTrue(path.endswith(self.MODEL_NAME))
