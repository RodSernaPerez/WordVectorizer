"""Class ModelDataLoader implementation"""
import json
import os
from typing import List

from word_vectorizer.models.model_data import ModelData


class ModelDataLoader:
    """Class that reads from disk the info about the models."""

    _NAME_FILE = 'data_models.json'
    _CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

    _PATH_TO_FILE = os.path.join(_CURRENT_PATH, _NAME_FILE)

    @classmethod
    def load_data(cls) -> List[ModelData]:
        """Loads the data of the available models from disk."""
        with open(cls._PATH_TO_FILE) as f:
            data_jsons = json.loads(f.read())

        return [ModelData(**data) for data in data_jsons]
