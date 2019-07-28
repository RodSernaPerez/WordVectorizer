from unittest import TestCase
from unittest.mock import patch, MagicMock

from word_vectorizer.models.model_data import ModelData
from word_vectorizer.models.model_data_loader import ModelDataLoader


class TestModelDataLoader(TestCase):
    @patch('word_vectorizer.models.model_data_loader.ModelData')
    def test_load_data_OK(self, mock_data_model):
        mock_data_model.return_value = MagicMock(ModelData)

        data_models = ModelDataLoader.load_data()
        self.assertTrue(isinstance(data_models, list))
        self.assertTrue(all([isinstance(x, ModelData) for x in data_models]))
