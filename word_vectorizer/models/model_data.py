"""ModelData implementation

Implements a class that manages the info associated to a model."""


class ModelData:
    """Data associated to a vectirization model"""

    def __init__(self, name: str, tech: str, dimensions: int, language: str,
                 url: str, binary: bool, description: str):
        self.name = name
        self.tech = tech
        self.dimensions = dimensions
        self.language = language
        self.url = url
        self.binary = binary
        self.description = description

    def __iter__(self):
        for k, v in self.__dict__.items():
            yield k, v
