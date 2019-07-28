"""Class with constants of the module."""
import os


class Constants:
    """Constants"""

    NAME_FOLDER_MODEL = ".vectorizers"
    """str: Name of the folder where models will be downloaded"""

    DESTINATION_FOLDER = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        NAME_FOLDER_MODEL)
    """str: Full path to the folder where models will be donwloaded."""
