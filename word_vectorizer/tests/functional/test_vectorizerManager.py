import os
import shutil
from unittest import TestCase

from word_vectorizer.constants import Constants
from word_vectorizer.exceptions.vector_not_computed_exception import \
    VectorNotComputedException

Constants.DESTINATION_FOLDER = os.path.join(Constants.DESTINATION_FOLDER,
                                            "../functional_test_models")
from word_vectorizer.vectorizer_manager import VectorizerManager  # noqa: E408


class TestVectorizerManager(TestCase):
    EXISTING_MODEL_NAME_DRIVE = "dummy_word_to_vec"
    WORD_IN_MODEL_DRIVE = "hola"
    WORD_NOT_IN_MODEL_DRIVE = "fmasdlkfmsad"
    EXISTING_MODEL_GENSIM = "glove-twitter-50"
    EXISTINNG_MODEL_NOT_DRIVE = "word2vec_SBWC_spanish_300.txt.bz2"

    def setUp(self) -> None:
        if os.path.exists(Constants.DESTINATION_FOLDER):
            shutil.rmtree(Constants.DESTINATION_FOLDER)

    def tearDown(self) -> None:
        if os.path.exists(Constants.DESTINATION_FOLDER):
            shutil.rmtree(Constants.DESTINATION_FOLDER)

    def test_functional_test_drive_OK(self):
        name_model = self.EXISTING_MODEL_NAME_DRIVE

        vm = VectorizerManager()
        print("Initial checks")
        self.assertTrue(name_model in vm.list_available_models())
        self.assertEqual(vm.list_loaded_models(), [])
        self.assertEqual(vm.list_downloaded_models(), [])

        print("Loading model")
        vm.load_model(name_model)

        self.assertTrue(name_model in vm.list_downloaded_models())
        self.assertTrue(name_model in vm.list_loaded_models())

        print("Use model")
        vm.vectorize(name_model, self.WORD_IN_MODEL_DRIVE)

        with self.assertRaises(VectorNotComputedException):
            vm.vectorize(name_model, self.WORD_NOT_IN_MODEL_DRIVE)

        print("Clear everything")
        vm.remove_all_models_from_disk()
        self.assertEqual(vm.list_downloaded_models(), [])
        vm.clear_loaded_models()
        self.assertEqual(vm.list_loaded_models(), [])

        del vm
