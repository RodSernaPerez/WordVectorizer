import os
from unittest import TestCase
from unittest.mock import MagicMock, patch

import gensim

from word_vectorizer.models.model_data import ModelData
from word_vectorizer.models.word2vec_vectorizer import Word2VecVectorizer


class TestWord2VecVectorizer(TestCase):
    FOLDER_FOR_MODELS = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), "testmodels")

    NAME_W2V_FILE_NON_BINARY = "test.non.binary.txt"
    NAME_W2V_FILE_BINARY = "test.binary.bin"

    PATH_TO_NON_BINARY_MODEL = os.path.join(FOLDER_FOR_MODELS,
                                            NAME_W2V_FILE_NON_BINARY)
    PATH_TO_BINARY_MODEL = os.path.join(FOLDER_FOR_MODELS,
                                        NAME_W2V_FILE_NON_BINARY)

    WORDS_IN_MODEL = ["hola", "adios"]
    SIZE_VECTORS = 4

    @patch(Word2VecVectorizer.__module__ + ".KeyedVectors")
    @patch(Word2VecVectorizer.__module__ + ".get_tmpfile")
    def test_create_model_non_binary_OK(self, mock_get_tempfile,
                                        mock_word2vec):
        mock_model_data = MagicMock(ModelData)
        mock_model_data.binary = False

        mock_word2vec.return_value = MagicMock(gensim.models.Word2Vec)

        w2v_model = Word2VecVectorizer(mock_model_data,
                                       self.PATH_TO_NON_BINARY_MODEL)

        mock_get_tempfile.assert_called_once_with(
            self.PATH_TO_NON_BINARY_MODEL)
        mock_word2vec.load_word2vec_format.assert_called_once_with(
            mock_get_tempfile.return_value,
            binary=mock_model_data.binary)

        self.assertEqual(mock_word2vec.load_word2vec_format.return_value,
                         w2v_model.model)

    @patch(Word2VecVectorizer.__module__ + ".KeyedVectors")
    @patch(Word2VecVectorizer.__module__ + ".get_tmpfile")
    def test_create_model_binary_OK(self, mock_get_tempfile,
                                    mock_word2vec):
        mock_model_data = MagicMock(ModelData)
        mock_model_data.binary = True

        mock_word2vec.return_value = MagicMock(gensim.models.Word2Vec)

        w2v_model = Word2VecVectorizer(mock_model_data,
                                       self.PATH_TO_BINARY_MODEL)

        mock_get_tempfile.assert_called_once_with(
            self.PATH_TO_BINARY_MODEL)
        mock_word2vec.load_word2vec_format.assert_called_once_with(
            mock_get_tempfile.return_value,
            binary=mock_model_data.binary)

        self.assertEqual(mock_word2vec.load_word2vec_format.return_value,
                         w2v_model.model)
