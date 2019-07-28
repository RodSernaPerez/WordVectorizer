import sys
from unittest import TestCase
from unittest.mock import MagicMock, patch

from word_vectorizer.exceptions.not_known_technology_exception \
    import NotKnownTechnologyException
from word_vectorizer.models.fasttext_vectorizer import FastTextVectorizer
from word_vectorizer.models.glove_vectorizer import GloveVectorizer
from word_vectorizer.models.model_data import ModelData
from word_vectorizer.models.word2vec_vectorizer import Word2VecVectorizer


class TestVectorizerBuilder(TestCase):
    TAG_FOR_WORD2VEC = "word2vec"
    TAG_FOR_GLOVE = "glove"
    TAG_FOR_FASTTEXT = "fasttext"

    UNKNOWN_TAG = "dnasjkdnalds"

    @classmethod
    def setUpClass(cls) -> None:
        """This weird thing must be done so VectorizerBuilder is imported
        after mocking."""
        cls.mock_w2v = patch(".".join([Word2VecVectorizer.__module__,
                                       Word2VecVectorizer.__name__])).start()
        cls.mock_glove = patch(".".join([GloveVectorizer.__module__,
                                         GloveVectorizer.__name__])).start()
        cls.mock_fasttext = patch(".".join(
            [FastTextVectorizer.__module__,
             FastTextVectorizer.__name__])).start()

        from word_vectorizer.models.vectorizer_builder import VectorizerBuilder
        del sys.modules[VectorizerBuilder.__module__]
        from word_vectorizer.models.vectorizer_builder import VectorizerBuilder

        cls.vectorizer_builder = VectorizerBuilder

    @classmethod
    def tearDownClass(cls) -> None:
        patch.stopall()

    def test_build_vectorizer_word2vec_OK(self):
        self.mock_w2v.return_value = MagicMock(Word2VecVectorizer)
        data_of_model = MagicMock(ModelData)
        data_of_model.tech = self.TAG_FOR_WORD2VEC

        vectorizer = \
            self.vectorizer_builder.build_vectorizer(data_of_model, "path")

        self.assertTrue(isinstance(vectorizer, Word2VecVectorizer))

    def test_build_vectorizer_glove_OK(self):
        self.mock_glove.return_value = MagicMock(GloveVectorizer)
        data_of_model = MagicMock(ModelData)
        data_of_model.tech = self.TAG_FOR_GLOVE

        vectorizer = \
            self.vectorizer_builder.build_vectorizer(data_of_model, "path")

        self.assertTrue(isinstance(vectorizer, GloveVectorizer))

    def test_build_vectorizer_fasttext_OK(self):
        self.mock_fasttext.return_value = MagicMock(FastTextVectorizer)
        data_of_model = MagicMock(ModelData)
        data_of_model.tech = self.TAG_FOR_FASTTEXT

        vectorizer = \
            self.vectorizer_builder.build_vectorizer(data_of_model, "path")

        self.assertTrue(isinstance(vectorizer, FastTextVectorizer))

    def test_build_vectorizer_not_good_tag_FAIL(self):
        data_of_model = MagicMock(ModelData)
        data_of_model.tech = self.UNKNOWN_TAG

        with self.assertRaises(NotKnownTechnologyException):
            self.vectorizer_builder.build_vectorizer(data_of_model, "path")
