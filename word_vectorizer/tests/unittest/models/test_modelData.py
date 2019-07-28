from unittest import TestCase

from word_vectorizer.models.model_data import ModelData


class TestModelData(TestCase):
    NAME_MODEL = 'word2vec_SBWC_spanish_300'
    TECH = 'word2vec'
    DIMENSIONS = 300
    LANGUAGE = 'spanish'
    URL = "http://cs.famaf.unc.edu.ar/~ccardellino/SBWCE/SBW-vectors-300" \
          "-min5.txt.bz2 "
    BINARY = False
    DESCRIPTION = "Word2Vec SBWC dim 300"

    def test___dict___OK(self):
        data = ModelData(self.NAME_MODEL, self.TECH, self.DIMENSIONS,
                         self.LANGUAGE, self.URL, self.BINARY,
                         self.DESCRIPTION)

        self.assertEqual(data.name, self.NAME_MODEL)
        self.assertEqual(data.tech, self.TECH)
        self.assertEqual(data.dimensions, self.DIMENSIONS)
        self.assertEqual(data.language, self.LANGUAGE)
        self.assertEqual(data.url, self.URL)
        self.assertEqual(data.binary, self.BINARY)
        self.assertEqual(data.description, self.DESCRIPTION)

        self.assertEqual(dict(data),
                         {"name": self.NAME_MODEL,
                          "tech": self.TECH,
                          "dimensions": self.DIMENSIONS,
                          "language": self.LANGUAGE,
                          "url": self.URL,
                          "binary": self.BINARY,
                          "description": self.DESCRIPTION})
