import os
import os.path as osp
import pickle  # nosec - disable B403:import_pickle check
import shutil

import numpy as np
import pytest
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
from datumaro.plugins.yolo_format.converter import YoloConverter
from datumaro.plugins.yolo_format.extractor import YoloExtractor
from datumaro.plugins.yolo_format.importer import YoloImporter
from datumaro.util.image import save_image
from datumaro.util.test_utils import TestDir, compare_datasets, compare_datasets_strict

from ...requirements import Requirements, mark_requirement
from ...utils.assets import get_test_asset_path


class YoloConverterTest:
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self, helper_tc):
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
                    subset="train",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                        Bbox(2, 1, 2, 3, label=4),
                    ],
                ),
                DatasetItem(
                    id=3,
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
            YoloConverter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(path="1.jpg", size=(10, 15)),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloConverter.convert(source_dataset, test_dir)

            save_image(
                osp.join(test_dir, "obj_train_data", "1.jpg"), np.ones((10, 15, 3))
            )  # put the image for dataset
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_dataset_with_exact_image_info(self, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(path="1.jpg", size=(10, 15)),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloConverter.convert(source_dataset, test_dir)

            parsed_dataset = Dataset.import_from(test_dir, "yolo", image_info={"1": (10, 15)})

            compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
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
            YoloConverter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(helper_tc, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("save_media", [True, False])
    def test_relative_paths(self, helper_tc, save_media):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", media=Image(data=np.ones((4, 2, 3)))),
                DatasetItem(id="subdir1/1", subset="train", media=Image(data=np.ones((2, 6, 3)))),
                DatasetItem(id="subdir2/1", subset="train", media=Image(data=np.ones((5, 4, 3)))),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            YoloConverter.convert(source_dataset, test_dir, save_media=save_media)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self, helper_tc):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "q/1", subset="train", media=Image(path="q/1.JPEG", data=np.zeros((4, 3, 3)))
                ),
                DatasetItem(
                    "a/b/c/2",
                    subset="valid",
                    media=Image(path="a/b/c/2.bmp", data=np.zeros((3, 4, 3))),
                ),
            ],
            categories=[],
        )

        with TestDir() as test_dir:
            YoloConverter.convert(dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(helper_tc, dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self, helper_tc):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image(data=np.ones((3, 2, 3)))),
            ],
            categories=[],
        )

        with TestDir() as path:
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(1, subset="train", media=Image(data=np.ones((2, 4, 3)))),
                    DatasetItem(2, subset="train", media=Image(path="2.jpg", size=(3, 2))),
                    DatasetItem(3, subset="valid", media=Image(data=np.ones((2, 2, 3)))),
                ],
                categories=[],
            )
            dataset.export(path, "yolo", save_media=True)

            dataset.put(DatasetItem(2, subset="train", media=Image(data=np.ones((3, 2, 3)))))
            dataset.remove(3, "valid")
            dataset.save(save_media=True)

            assert set(os.listdir(osp.join(path, "obj_train_data"))) == {
                "1.txt",
                "2.txt",
                "1.jpg",
                "2.jpg",
            }
            assert set(os.listdir(osp.join(path, "obj_valid_data"))) == set()
            compare_datasets(
                helper_tc, expected, Dataset.import_from(path, "yolo"), require_media=True
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_meta_file(self, helper_tc):
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
                    subset="train",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                        Bbox(2, 1, 2, 3, label=4),
                    ],
                ),
                DatasetItem(
                    id=3,
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
            YoloConverter.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            assert osp.isfile(osp.join(test_dir, "dataset_meta.json"))
            compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_565)
    def test_can_save_and_load_with_custom_subset_name(self, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="anything",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=2),
                        Bbox(0, 2, 3, 2, label=5),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            YoloConverter.convert(source_dataset, test_dir, save_media=True)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_565)
    def test_cant_save_with_reserved_subset_name(self):
        for subset in ["backup", "classes"]:
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id=3,
                        subset=subset,
                        media=Image(data=np.ones((8, 8, 3))),
                    ),
                ],
                categories=["a"],
            )

            with TestDir() as test_dir:
                with pytest.raises(DatasetExportError, match=f"Can't export '{subset}' subset"):
                    YoloConverter.convert(dataset, test_dir)

    @mark_requirement(Requirements.DATUM_609)
    def test_can_save_and_load_without_path_prefix(self, helper_tc):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(0, 1, 5, 2, label=1),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        with TestDir() as test_dir:
            YoloConverter.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False)
            parsed_dataset = Dataset.import_from(test_dir, "yolo")

            with open(osp.join(test_dir, "obj.data"), "r") as f:
                lines = f.readlines()
                assert "valid = valid.txt\n" in lines

            with open(osp.join(test_dir, "valid.txt"), "r") as f:
                lines = f.readlines()
                assert "obj_valid_data/3.jpg\n" in lines

            compare_datasets(helper_tc, source_dataset, parsed_dataset)


DUMMY_DATASET_DIR = get_test_asset_path("yolo_dataset")


class YoloImporterTest:
    def test_can_detect(self):
        detected_formats = Environment().detect_dataset(DUMMY_DATASET_DIR)
        assert detected_formats == [YoloImporter.NAME]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(0, 2, 4, 2, label=2),
                        Bbox(3, 3, 2, 3, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        dataset = Dataset.import_from(DUMMY_DATASET_DIR, "yolo")

        compare_datasets(helper_tc, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_with_exif_rotated_images(self, helper_tc):
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((15, 10, 3))),
                    annotations=[
                        Bbox(0, 3, 2.67, 3.0, label=2),
                        Bbox(2, 4.5, 1.33, 4.5, label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        with TestDir() as test_dir:
            dataset_path = osp.join(test_dir, "dataset")
            shutil.copytree(DUMMY_DATASET_DIR, dataset_path)

            # Add exif rotation for image
            image_path = osp.join(dataset_path, "obj_train_data", "1.jpg")
            img = PILImage.open(image_path)
            exif = img.getexif()
            exif.update([(296, 3), (282, 28.0), (531, 1), (274, 6), (283, 28.0)])
            img.save(image_path, exif=exif)

            dataset = Dataset.import_from(dataset_path, "yolo")

            compare_datasets(helper_tc, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self, helper_tc):
        source = Dataset.import_from(DUMMY_DATASET_DIR, format="yolo")

        parsed = pickle.loads(pickle.dumps(source))  # nosec

        compare_datasets_strict(helper_tc, source, parsed)


class YoloExtractorTest:
    def _prepare_dataset(self, path: str) -> Dataset:
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "a",
                    subset="train",
                    media=Image(np.ones((5, 10, 3))),
                    annotations=[Bbox(1, 1, 2, 4, label=0)],
                )
            ],
            categories=["test"],
        )
        dataset.export(path, "yolo", save_media=True)

        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_parse(self, helper_tc):
        with TestDir() as test_dir:
            expected = self._prepare_dataset(test_dir)

            actual = Dataset.import_from(test_dir, "yolo")
            compare_datasets(helper_tc, expected, actual)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_data_file(self):
        with TestDir() as test_dir:
            with pytest.raises(DatasetImportError, match="Can't read dataset descriptor file"):
                YoloExtractor(test_dir)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_ann_line_format(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            with open(osp.join(test_dir, "obj_train_data", "a.txt"), "w") as f:
                f.write("1 2 3\n")

            with pytest.raises(AnnotationImportError) as capture:
                Dataset.import_from(test_dir, "yolo").init_cache()
            assert isinstance(capture.value.__cause__, InvalidAnnotationError)
            assert "Unexpected field count" in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_label(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            with open(osp.join(test_dir, "obj_train_data", "a.txt"), "w") as f:
                f.write("10 0.5 0.5 0.5 0.5\n")

            with pytest.raises(AnnotationImportError) as capture:
                Dataset.import_from(test_dir, "yolo").init_cache()
            assert isinstance(capture.value.__cause__, UndeclaredLabelError)
            assert capture.value.__cause__.id == "10"

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "field, field_name",
        [
            (1, "bbox center x"),
            (2, "bbox center y"),
            (3, "bbox width"),
            (4, "bbox height"),
        ],
    )
    def test_can_report_invalid_field_type(self, field, field_name):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            with open(osp.join(test_dir, "obj_train_data", "a.txt"), "w") as f:
                values = [0, 0.5, 0.5, 0.5, 0.5]
                values[field] = "a"
                f.write(" ".join(str(v) for v in values))

            with pytest.raises(AnnotationImportError) as capture:
                Dataset.import_from(test_dir, "yolo").init_cache()
            assert isinstance(capture.value.__cause__, InvalidAnnotationError)
            assert field_name in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_ann_file(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            os.remove(osp.join(test_dir, "obj_train_data", "a.txt"))

            with pytest.raises(ItemImportError) as capture:
                Dataset.import_from(test_dir, "yolo").init_cache()
            assert isinstance(capture.value.__cause__, FileNotFoundError)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_image_info(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            os.remove(osp.join(test_dir, "obj_train_data", "a.jpg"))

            with pytest.raises(ItemImportError) as capture:
                Dataset.import_from(test_dir, "yolo").init_cache()
            assert isinstance(capture.value.__cause__, DatasetImportError)
            assert "Can't find image info" in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_subset_info(self):
        with TestDir() as test_dir:
            self._prepare_dataset(test_dir)
            os.remove(osp.join(test_dir, "train.txt"))

            with pytest.raises(InvalidAnnotationError, match="subset list file"):
                Dataset.import_from(test_dir, "yolo").init_cache()
