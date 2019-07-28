import os
import pickle
import shutil
from unittest import TestCase
from unittest.mock import MagicMock, patch

import gensim
import numpy as np

from word_vectorizer.models.glove_vectorizer import GloveVectorizer
from word_vectorizer.models.model_data import ModelData


class TestGloveVectorizer(TestCase):
    FOLDER_FOR_MODELS = os.path.join(os.path.dirname(os.path.realpath(
        __file__)), "testmodels")

    NAME_GLOVE_FILE_NON_BINARY = "test.non.binary.txt"
    NAME_GLOVE_FILE_BINARY = "test.binary.bin"

    PATH_TO_NON_BINARY_MODEL = os.path.join(FOLDER_FOR_MODELS,
                                            NAME_GLOVE_FILE_NON_BINARY)
    PATH_TO_BINARY_MODEL = os.path.join(FOLDER_FOR_MODELS,
                                        NAME_GLOVE_FILE_NON_BINARY)

    WORDS_IN_MODEL = ["hola", "adios"]
    SIZE_VECTORS = 4

    def setUp(self) -> None:
        self.DICT_OF_VECTORS = {k: np.random.rand(self.SIZE_VECTORS) for k in
                                self.WORDS_IN_MODEL}
        if not os.path.exists(self.FOLDER_FOR_MODELS):
            os.makedirs(self.FOLDER_FOR_MODELS)

        # Creates a glove file
        word_vec_file = ""

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

    @patch(GloveVectorizer.__module__ + ".KeyedVectors")
    @patch(GloveVectorizer.__module__ + ".get_tmpfile")
    def test_create_model_non_binary_word2vec_format_OK(self,
                                                        mock_get_tempfile,
                                                        mock_glove):
        mock_model_data = MagicMock(ModelData)
        mock_model_data.binary = False

        mock_glove.return_value = MagicMock(gensim.models.Word2Vec)

        fast_text_model = GloveVectorizer(mock_model_data,
                                          self.PATH_TO_NON_BINARY_MODEL)

        mock_get_tempfile.assert_called_once_with(
            self.PATH_TO_NON_BINARY_MODEL)
        mock_glove.load_word2vec_format.assert_called_once_with(
            mock_get_tempfile.return_value,
            binary=mock_model_data.binary)

        self.assertEqual(mock_glove.load_word2vec_format.return_value,
                         fast_text_model.model)

    @patch(GloveVectorizer.__module__ + ".glove2word2vec")
    @patch(GloveVectorizer.__module__ + ".KeyedVectors")
    @patch(GloveVectorizer.__module__ + ".get_tmpfile")
    def test_create_model_non_binary_glove_format_OK(self,
                                                     mock_get_tempfile,
                                                     mock_KeyedVectors,
                                                     mock_glove2w2v):
        mock_model_data = MagicMock(ModelData)
        mock_model_data.binary = False

        model_to_return = MagicMock(gensim.models.Word2Vec)
        mock_KeyedVectors.load_word2vec_format.side_effect = \
            [ValueError,
             model_to_return]

        glove_model = GloveVectorizer(mock_model_data,
                                      self.PATH_TO_NON_BINARY_MODEL)

        mock_glove2w2v.assert_called_once_with(mock_get_tempfile.return_value,
                                               mock_get_tempfile.return_value)

        self.assertEqual(model_to_return, glove_model.model)

    @patch(GloveVectorizer.__module__ + ".KeyedVectors")
    @patch(GloveVectorizer.__module__ + ".get_tmpfile")
    def test_create_model_binary_word2vec_format_OK(self,
                                                    mock_get_tempfile,
                                                    mock_glove):
        mock_model_data = MagicMock(ModelData)
        mock_model_data.binary = True

        mock_glove.return_value = MagicMock(gensim.models.Word2Vec)

        fast_text_model = GloveVectorizer(mock_model_data,
                                          self.PATH_TO_BINARY_MODEL)

        mock_get_tempfile.assert_called_once_with(
            self.PATH_TO_BINARY_MODEL)
        mock_glove.load_word2vec_format.assert_called_once_with(
            mock_get_tempfile.return_value,
            binary=mock_model_data.binary)

        self.assertEqual(mock_glove.load_word2vec_format.return_value,
                         fast_text_model.model)

    @patch(GloveVectorizer.__module__ + ".glove2word2vec")
    @patch(GloveVectorizer.__module__ + ".KeyedVectors")
    @patch(GloveVectorizer.__module__ + ".get_tmpfile")
    def test_create_model_binary_glove_format_OK(self,
                                                 mock_get_tempfile,
                                                 mock_KeyedVectors,
                                                 mock_glove2w2v):
        mock_model_data = MagicMock(ModelData)
        mock_model_data.binary = False

        model_to_return = MagicMock(gensim.models.Word2Vec)
        mock_KeyedVectors.load_word2vec_format.side_effect = \
            [ValueError, model_to_return]

        glove_model = GloveVectorizer(mock_model_data,
                                      self.PATH_TO_BINARY_MODEL)

        mock_glove2w2v.assert_called_once_with(mock_get_tempfile.return_value,
                                               mock_get_tempfile.return_value)

        self.assertEqual(model_to_return, glove_model.model)
