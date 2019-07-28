import os
import pickle
import shutil
from unittest import TestCase
from unittest.mock import MagicMock, patch

import gensim
import numpy as np

from word_vectorizer.models.fasttext_vectorizer import FastTextVectorizer
from word_vectorizer.models.model_data import ModelData


class TestWFastTextVectorizer(TestCase):
    FOLDER_FOR_MODELS = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), "testmodels")

    NAME_FASTTEXT_FILE_NON_BINARY = "test.non.binary.txt"
    NAME_FASTTEXT_FILE_BINARY = "test.binary.bin"

    PATH_TO_NON_BINARY_MODEL = os.path.join(FOLDER_FOR_MODELS,
                                            NAME_FASTTEXT_FILE_NON_BINARY)
    PATH_TO_BINARY_MODEL = os.path.join(FOLDER_FOR_MODELS,
                                        NAME_FASTTEXT_FILE_NON_BINARY)

    WORDS_IN_MODEL = ["hola", "adios"]
    SIZE_VECTORS = 4

    def setUp(self) -> None:
        self.DICT_OF_VECTORS = {k: np.random.rand(self.SIZE_VECTORS) for k in
                                self.WORDS_IN_MODEL}
        if not os.path.exists(self.FOLDER_FOR_MODELS):
            os.makedirs(self.FOLDER_FOR_MODELS)

        # Creates a Word2Vec file
        word_vec_file = "{0} {1} \n".format(
            len(self.WORDS_IN_MODEL), self.SIZE_VECTORS)

        for w in self.WORDS_IN_MODEL:
            word_vec_file += (w + " ")
            for x in self.DICT_OF_VECTORS[w].tolist():
                word_vec_file += "{} ".format(x)
            word_vec_file += "\n"

        with open(self.PATH_TO_NON_BINARY_MODEL, "w") as f:
            f.write(word_vec_file)

        with open(self.PATH_TO_BINARY_MODEL, "wb") as f:
            pickle.dump(word_vec_file, f)

    def setDown(self):
        shutil.rmtree(self.FOLDER_FOR_MODELS, ignore_errors=True)

    @patch(FastTextVectorizer.__module__ + ".FastText")
    @patch(FastTextVectorizer.__module__ + ".get_tmpfile")
    def test_create_model_non_binary_OK(self, mock_get_tempfile,
                                        mock_fasttext):
        mock_model_data = MagicMock(ModelData)
        mock_model_data.binary = False

        mock_fasttext.return_value = MagicMock(gensim.models.FastText)

        fast_text_model = FastTextVectorizer(mock_model_data,
                                             self.PATH_TO_NON_BINARY_MODEL)

        mock_get_tempfile.assert_called_once_with(
            self.PATH_TO_NON_BINARY_MODEL)
        mock_fasttext.load.assert_called_once_with(
            mock_get_tempfile.return_value,
            mock_model_data.binary)

        self.assertEqual(mock_fasttext.load.return_value,
                         fast_text_model.model)

    @patch(FastTextVectorizer.__module__ + ".FastText")
    @patch(FastTextVectorizer.__module__ + ".get_tmpfile")
    def test_create_model_binary_OK(self, mock_get_tempfile,
                                    mock_fasttext):
        mock_model_data = MagicMock(ModelData)
        mock_model_data.binary = True

        mock_fasttext.return_value = MagicMock(gensim.models.FastText)

        fast_text_model = FastTextVectorizer(mock_model_data,
                                             self.PATH_TO_BINARY_MODEL)

        mock_get_tempfile.assert_called_once_with(
            self.PATH_TO_BINARY_MODEL)
        mock_fasttext.load.assert_called_once_with(
            mock_get_tempfile.return_value,
            mock_model_data.binary)

        self.assertEqual(mock_fasttext.load.return_value,
                         fast_text_model.model)
