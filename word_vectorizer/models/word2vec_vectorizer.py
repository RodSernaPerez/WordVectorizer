import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.test.utils import get_tmpfile

from word_vectorizer.models.non_contextual_vectorizer import \
    NonContextualVectorizer


class Word2VecVectorizer(NonContextualVectorizer):

    def _load_model(self, path_to_model: str):
        if "/" not in path_to_model:
            return api.load(path_to_model)
        tmp_file = get_tmpfile(path_to_model)
        return KeyedVectors.load_word2vec_format(tmp_file,
                                                 binary=self.model_data.binary)
