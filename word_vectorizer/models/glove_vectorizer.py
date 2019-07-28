import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import get_tmpfile

from word_vectorizer.models.non_contextual_vectorizer import \
    NonContextualVectorizer


class GloveVectorizer(NonContextualVectorizer):

    def _load_model(self, path_to_model: str):
        if "/" not in path_to_model:
            return api.load(path_to_model)
        path = get_tmpfile(path_to_model)

        try:
            model = self._load_as_word2vec_format(path)
        except ValueError:
            self._convert_format(path)
            model = self._load_as_word2vec_format(path)

        return model

    def _load_as_word2vec_format(self, path):
        return KeyedVectors.load_word2vec_format(path,
                                                 binary=self.model_data.binary)

    def _convert_format(self, path):
        glove2word2vec(path, path)
