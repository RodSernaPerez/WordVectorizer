import gensim.downloader as api
from gensim.models import FastText
from gensim.test.utils import get_tmpfile

from word_vectorizer.models.non_contextual_vectorizer import \
    NonContextualVectorizer


class FastTextVectorizer(NonContextualVectorizer):
    def _load_model(self, path_to_model: str):
        if "/" not in path_to_model:
            return api.load(path_to_model)
        tmp_file = get_tmpfile(path_to_model)
        return FastText.load(tmp_file, self.model_data.binary)
