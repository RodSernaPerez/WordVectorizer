"""Exceptions that can be launched when using this package"""
from .not_existing_model_exception import NotExistingModelException
from .not_known_technology_exception import NotKnownTechnologyException
from .not_understood_url_exception import NotUnderstoodURLException
from .vector_not_computed_exception import VectorNotComputedException

__all__ = [NotExistingModelException.__name__,
           NotKnownTechnologyException.__name__,
           NotUnderstoodURLException.__name__,
           VectorNotComputedException.__name__]
