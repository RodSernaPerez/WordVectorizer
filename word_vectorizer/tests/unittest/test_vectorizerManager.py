import os
import shutil
from unittest import TestCase
from unittest.mock import patch, MagicMock

import numpy as np

from word_vectorizer.constants import Constants
from word_vectorizer.exceptions.not_existing_model_exception import \
    NotExistingModelException
from word_vectorizer.model_downloading.model_downloader import ModelDownloader
from word_vectorizer.models.model_data import ModelData
from word_vectorizer.models.model_data_loader import ModelDataLoader
from word_vectorizer.models.vectorizer import Vectorizer
from word_vectorizer.vectorizer_manager import VectorizerManager


class TestVectorizerManager(TestCase):
    PATH_FOR_TESTING_MODELS = os.path.join(Constants.DESTINATION_FOLDER,
                                           "../test_models")

    def setUp(self) -> None:
        if not os.path.exists(self.PATH_FOR_TESTING_MODELS):
            os.makedirs(self.PATH_FOR_TESTING_MODELS)
        Constants.DESTINATION_FOLDER = self.PATH_FOR_TESTING_MODELS

        patch(VectorizerManager.__module__ + "." +
              ModelDataLoader.__name__).start()

    def tearDown(self):
        patch.stopall()
        if os.path.exists(self.PATH_FOR_TESTING_MODELS):
            shutil.rmtree(self.PATH_FOR_TESTING_MODELS)

    @patch(VectorizerManager.__module__ + ".ModelDataLoader.load_data")
    def test_load_data_of_models_OK(self, mock_load_data):
        data_models = [MagicMock(ModelData) for _ in range(3)]
        mock_load_data.return_value = data_models
        vm = VectorizerManager()
        vm.load_data_of_models()
        self.assertEqual(vm.models_parameters, data_models)

    @patch(VectorizerManager.__module__ + '.listdir')
    @patch(VectorizerManager.__module__ + '.isfile')
    def test_list_downloaded_models_OK(self, mock_is_file, mock_listdir):
        models_in_disk = ["model_1", "model_2", "model_3"]
        mock_is_file.return_value = True
        mock_listdir.return_value = models_in_disk

        vm = VectorizerManager()
        self.assertEqual(models_in_disk, vm.list_downloaded_models())

    def test_list_loaded_models_OK(self):
        model_names = list({"model_1", "model_2", "model_3"})
        vm = VectorizerManager()
        vm.loaded_models = {k: MagicMock(Vectorizer) for k in model_names}

        self.assertEqual(vm.list_loaded_models().sort(),
                         model_names.sort())

    @patch(VectorizerManager.__module__ +
           ".ModelDownloaderGetter.get_downloader")
    @patch(VectorizerManager.__module__ +
           ".VectorizerBuilder.build_vectorizer")
    def test_load_model_not_loaded_model_OK(self,
                                            mock_vec_builder,
                                            mock_get_downloader):
        vm = VectorizerManager()
        data_model_1 = MagicMock(ModelData)
        data_model_1.name = "model_1"
        data_model_1.url = "an_url"
        data_model_2 = MagicMock(ModelData)
        data_model_2.name = "model_2"

        vm.models_parameters = [data_model_1, data_model_2]

        path = "a/path/to/model"

        downloader = MagicMock(ModelDownloader)
        downloader.download_from_url.return_value = path
        mock_get_downloader.return_value = downloader

        vectorizer = MagicMock(Vectorizer)
        mock_vec_builder.return_value = vectorizer

        asked_model = data_model_1.name
        vm.load_model(asked_model)

        self.assertEqual(vm.loaded_models[asked_model], vectorizer)
        mock_vec_builder.assert_called_once_with(data_model_1, path)
        mock_get_downloader.assert_called_once_with(data_model_1.url)
        downloader.download_from_url.assert_called_once_with(data_model_1.url,
                                                             data_model_1.name)

    @patch(VectorizerManager.__module__ +
           ".ModelDownloaderGetter.get_downloader")
    @patch(VectorizerManager.__module__ +
           ".VectorizerBuilder.build_vectorizer")
    def test_load_model_loaded_model_OK(self,
                                        mock_vec_builder,
                                        mock_get_downloader):
        vm = VectorizerManager()
        data_model_1 = MagicMock(ModelData)
        data_model_1.name = "model_1"
        data_model_1.url = "an_url"
        data_model_2 = MagicMock(ModelData)
        data_model_2.name = "model_2"

        vm.models_parameters = [data_model_1, data_model_2]

        downloader = MagicMock(ModelDownloader)
        mock_get_downloader.return_value = downloader

        vectorizer = MagicMock(Vectorizer)
        vm.loaded_models = {data_model_1.name: vectorizer}

        asked_model = data_model_1.name
        vm.load_model(asked_model)

        self.assertFalse(mock_vec_builder.called)
        self.assertFalse(mock_get_downloader.called)
        self.assertFalse(downloader.download_from_url.called)

        self.assertEqual(vm.loaded_models[asked_model], vectorizer)

    def test_load_model_not_existing_model_FAIL(self):
        vm = VectorizerManager()

        with self.assertRaises(NotExistingModelException):
            vm.load_model("XXXXX")

    @patch(VectorizerManager.__module__ + ".VectorizerManager.load_model")
    def test_vectorize_not_loaded_model_OK(self, mock_load_model):
        name_model = "model_x"
        word = "word"
        vector = np.zeros(20)
        vm = VectorizerManager()
        vectorizer = MagicMock(Vectorizer)
        vectorizer.return_value = vector

        def mock_loading(_):
            vm.loaded_models = {name_model: vectorizer}

        mock_load_model.side_effect = mock_loading

        self.assertTrue(np.array_equal(vm.vectorize(name_model, word), vector))

        mock_load_model.assert_called_once_with(name_model)

    @patch(VectorizerManager.__module__ + ".VectorizerManager.load_model")
    def test_vectorize_loaded_model_OK(self, mock_load_model):
        name_model = "model_x"
        word = "word"
        vector = np.zeros(20)
        vm = VectorizerManager()
        vectorizer = MagicMock(Vectorizer)
        vectorizer.return_value = vector

        vm.loaded_models = {name_model: vectorizer}

        self.assertTrue(np.array_equal(vm.vectorize(name_model, word), vector))

        self.assertFalse(mock_load_model.called)

    @patch(VectorizerManager.__module__ + ".VectorizerManager.load_model")
    def test_vectorize_not_existing_model_FAIL(self, mock_load_model):
        vm = VectorizerManager()

        mock_load_model.side_effect = NotExistingModelException

        with self.assertRaises(NotExistingModelException):
            vm.vectorize("XXXX", "word")

    def test_clear_loaded_models_OK(self):
        vm = VectorizerManager()
        vm.loaded_models = {"model": MagicMock(Vectorizer)}

        vm.clear_loaded_models()
        self.assertEqual({}, vm.loaded_models)

    def test_remove_model_OK(self):
        model = "model"
        model_2 = "model_2"

        vm = VectorizerManager()
        vm.loaded_models = {model: MagicMock(Vectorizer),
                            model_2: MagicMock(Vectorizer)}

        vm.remove_model(model)

        self.assertFalse(model in vm.loaded_models.keys())
        self.assertTrue(model_2 in vm.loaded_models.keys())

    def test_remove_model_not_existing_FAIL(self):
        model = "model"
        model_2 = "model_2"

        vm = VectorizerManager()
        models_in_memory = {model: MagicMock(Vectorizer),
                            model_2: MagicMock(Vectorizer)}
        vm.loaded_models = models_in_memory

        non_existing_model = "xxxxxxx"

        with self.assertRaises(NotExistingModelException):
            vm.remove_model(non_existing_model)

        self.assertEqual(vm.loaded_models, models_in_memory)

    def test_remove_all_models_from_disk_OK(self):
        vm = VectorizerManager()
        vm.remove_all_models_from_disk()
        self.assertFalse(os.path.exists(self.PATH_FOR_TESTING_MODELS))

    def test_list_available_models_OK(self):
        vm = VectorizerManager()

        names_models = ["model_1", "model_2", "model_3", "model_4"]

        model_parameters = []
        for n in names_models:
            x = MagicMock(ModelData)
            x.name = n
            model_parameters.append(x)
        vm.models_parameters = model_parameters

        self.assertEqual(sorted(vm.list_available_models()),
                         sorted(names_models))

    def test_describe_model_OK(self):
        vm = VectorizerManager()

        names_models = ["model_1", "model_2", "model_3", "model_4"]
        models = [MagicMock(ModelData) for _ in names_models]
        info_of_models = [{"id": np.random.randint(100)} for _ in names_models]

        def dict_to_iterator(info):
            for k, v in info.items():
                yield k, v

        model_parameters = []
        for i in range(len(models)):
            x = models[i]
            x.__iter__.return_value = dict_to_iterator(info_of_models[i])
            x.name = names_models[i]
            model_parameters.append(x)
        vm.models_parameters = model_parameters

        description = vm.describe_model(names_models[0])

        self.assertEqual(description, info_of_models[0])

        for x in models[1:]:
            self.assertFalse(x.called)

    def test_describe_model_not_existing_model_FAIL(self):
        vm = VectorizerManager()

        names_models = ["model_1", "model_2", "model_3", "model_4"]
        not_a_model = "model_x"
        models = [MagicMock(ModelData) for _ in names_models]
        info_of_models = [{"id": np.random.randint(100)} for _ in names_models]

        def dict_to_iterator(info):
            for k, v in info.items():
                yield k, v

        model_parameters = []
        for i in range(len(models)):
            x = models[i]
            x.__iter__.return_value = dict_to_iterator(info_of_models[i])
            x.name = names_models[i]
            model_parameters.append(x)
        vm.models_parameters = model_parameters

        with self.assertRaises(NotExistingModelException):
            vm.describe_model(not_a_model)
