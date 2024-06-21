import os
import os.path as osp
import pickle  # nosec - disable B403:import_pickle check
import shutil
from unittest import TestCase

import numpy as np
from PIL import Image as PILImage

from datumaro.components.annotation import Bbox
from datumaro.components.dataset import Dataset
from datumaro.components.environment import Environment
from datumaro.components.errors import (
    AnnotationImportError,
    DatasetExportError,
    DatasetImportError,
    InvalidAnnotationError,
    ItemImportError,
    UndeclaredLabelError,
)
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.plugins.yolo_detection_format.converter import YoloDetectionConverter
from datumaro.plugins.yolo_detection_format.extractor import YoloDetectionExtractor
from datumaro.plugins.yolo_detection_format.importer import YoloDetectionImporter
from datumaro.util.image import save_image
from datumaro.util.test_utils import TestDir, compare_datasets, compare_datasets_strict

from .requirements import Requirements, mark_requirement


class YoloDetectionConvertertTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(0, 1, 2, 3, label=4),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=2),
                        Bbox(0, 2, 3, 2, label=5),
                        Bbox(0, 2, 4, 2, label=6),
                        Bbox(0, 7, 3, 2, label=7),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloDetectionConverter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo_detection")

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(                    
                    id=1,
                    subset="train",
                    media=Image(path="1.jpg", size=(10, 15)),
                    annotations=[
                        Bbox(0, 2, 4, 1, label=2, id=0),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloDetectionConverter.convert(source_dataset, test_dir)

            save_image(
                osp.join(test_dir, "images", "train", "1.jpg"), np.ones((10, 15, 3))
            )  # put the image for dataset
            parsed_dataset = Dataset.import_from(test_dir, "yolo_detection")

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица c пробелом",
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(0, 1, 2, 3, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloDetectionConverter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo_detection")

            compare_datasets(self, source_dataset, parsed_dataset, require_media=True)


    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image(path="1.JPEG", data=np.zeros((4, 3, 3))), 
                    annotations=[
                        Bbox(0, 2, 3, 2, label=2),
                    ],
                ),
                DatasetItem(
                    id="2",
                    subset="valid",
                    media=Image(path="2.bmp", data=np.zeros((3, 4, 3))), 
                    annotations=[
                        Bbox(0, 1, 5, 2, label=2),
                    ]
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloDetectionConverter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo_detection")

            compare_datasets(self, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_meta_file(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(0, 1, 2, 3, label=4),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=2),
                        Bbox(0, 2, 3, 2, label=5),
                        Bbox(0, 2, 4, 2, label=6),
                        Bbox(0, 7, 3, 2, label=7),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloDetectionConverter.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo_detection")

            self.assertTrue(osp.isfile(osp.join(test_dir, "dataset_meta.json")))
            compare_datasets(self, source_dataset, parsed_dataset)


    @mark_requirement(Requirements.DATUM_609)
    def test_can_save_and_load_without_path_prefix(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=1),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        with TestDir() as test_dir:
            YoloDetectionConverter.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False)
            parsed_dataset = Dataset.import_from(test_dir, "yolo_detection")

            with open(osp.join(test_dir, "data.yaml"), "r") as f:
                lines = f.readlines()
                self.assertIn("train: train.txt\n", lines)

            with open(osp.join(test_dir, "train.txt"), "r") as f:
                lines = f.readlines()
                self.assertIn("./images/train/3.jpg\n", lines)

            compare_datasets(self, source_dataset, parsed_dataset)


DUMMY_DATASET_DIR = osp.join(osp.dirname(__file__), "assets", "yolo_detection_dataset")


class YoloDetectionImporterTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(0, 3, 14, 5, label=1),
                        Bbox(7, 0, 7, 4, label=0),
                    ],
                ),
            ],
            categories=["person", "bicycle"],
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "yolo_detection")

        compare_datasets(self, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_with_exif_rotated_images(self):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(0, 3, 14, 5, label=1),
                        Bbox(7, 0, 7, 4, label=0),
                    ],
                ),
            ],
            categories=["person", "bicycle"],
        )

        with TestDir() as test_dir:
            dataset_path = osp.join(test_dir, "dataset")
            shutil.copytree(DUMMY_DATASET_DIR, dataset_path)

            # Add exif rotation for image
            image_path = osp.join(dataset_path, "images", "train", "1.jpg")
            img = PILImage.open(image_path)
            exif = img.getexif()
            exif.update([(296, 3), (282, 28.0), (531, 1), (274, 6), (283, 28.0)])
            img.save(image_path, exif=exif)

            dataset = Dataset.import_from(dataset_path, "yolo_detection")

            compare_datasets(self, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self):
        source = Dataset.import_from(DUMMY_DATASET_DIR, format="yolo_detection")

        parsed = pickle.loads(pickle.dumps(source))  # nosec

        compare_datasets_strict(self, source, parsed)


class YoloDetectionExtractorTest(TestCase):
    def _prepare_dataset(self, path: str) -> Dataset:
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id = "a",
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[Bbox(0, 2, 4, 2, label=0)],
                )
            ],
            categories=["test"],
        )
        dataset.export(path, "yolo_detection", save_images=True)

        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_parse(self):
        with TestDir() as test_dir:
            expected = self._prepare_dataset(test_dir)

            actual = Dataset.import_from(test_dir, "yolo_detection")
            compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_data_file(self):
        with TestDir() as test_dir:
            with self.assertRaisesRegex(DatasetImportError, f"Can't find data.yaml in {test_dir}"):
                YoloDetectionExtractor(test_dir)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_ann_line_format(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            with open(osp.join(test_dir, "labels", "train", "a.txt"), "w") as f:
                f.write("1 2 3\n")

            with self.assertRaises(AnnotationImportError) as capture:
                Dataset.import_from(test_dir, "yolo_detection").init_cache()
            self.assertIsInstance(capture.exception.__cause__, InvalidAnnotationError)
            self.assertIn("Unexpected field count", str(capture.exception.__cause__))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_label(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            with open(osp.join(test_dir, "labels", "train", "a.txt"), "w") as f:
                f.write("10 0.5 0.5 0.5 0.5\n")

            with self.assertRaises(AnnotationImportError) as capture:
                Dataset.import_from(test_dir, "yolo_detection").init_cache()
            self.assertIsInstance(capture.exception.__cause__, UndeclaredLabelError)
            self.assertEqual(capture.exception.__cause__.id, "10")

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_field_type(self):
        for field, field_name in [
            (1, "bbox center x"),
            (2, "bbox center y"),
            (3, "bbox width"),
            (4, "bbox height"),
        ]:
            with self.subTest(field_name=field_name):
                with TestDir() as test_dir:
                    self._prepare_dataset(test_dir)
                    with open(osp.join(test_dir, "labels", "train", "a.txt"), "w") as f:
                        values = [0, 0.5, 0.5, 0.5, 0.5]
                        values[field] = "a"
                        f.write(" ".join(str(v) for v in values))

                    with self.assertRaises(AnnotationImportError) as capture:
                        Dataset.import_from(test_dir, "yolo_detection").init_cache()
                    self.assertIsInstance(capture.exception.__cause__, InvalidAnnotationError)
                    self.assertIn(field_name, str(capture.exception.__cause__))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_image_info(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            os.remove(osp.join(test_dir, "images", "train", "a.jpg"))

            with self.assertRaises(ItemImportError) as capture:
                Dataset.import_from(test_dir, "yolo_detection").init_cache()

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_subset_info(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            os.remove(osp.join(test_dir, "train.txt"))

            with self.assertRaisesRegex(InvalidAnnotationError, "subset list file"):
                Dataset.import_from(test_dir, "yolo_detection").init_cache()
