import os
import os.path as osp
import pickle  # nosec - disable B403:import_pickle check
import random
import shutil

import numpy as np
import pytest
import yaml
from PIL import Image as PILImage

from datumaro.components.annotation import Bbox, Polygon
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
from datumaro.plugins.yolo_format.converter import (
    Yolo8Converter,
    Yolo8SegmentationConverter,
    YoloConverter,
)
from datumaro.plugins.yolo_format.extractor import Yolo8Extractor, YoloExtractor
from datumaro.plugins.yolo_format.importer import YoloImporter
from datumaro.util.image import save_image
from datumaro.util.test_utils import compare_datasets, compare_datasets_strict

from ...requirements import Requirements, mark_requirement
from ...utils.assets import get_test_asset_path


class YoloConverterTest:
    CONVERTER = YoloConverter

    def _generate_random_annotation(self, n_of_labels=10):
        return Bbox(
            x=random.randint(0, 4),
            y=random.randint(0, 4),
            w=random.randint(1, 4),
            h=random.randint(1, 4),
            label=random.randint(0, n_of_labels - 1),
        )

    @staticmethod
    def _make_image_path(test_dir: str, subset_name: str, image_id: str):
        return osp.join(test_dir, f"obj_{subset_name}_data", image_id)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self, helper_tc, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, "yolo")

        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self, helper_tc, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(path="1.jpg", size=(10, 15)),
                    annotations=[
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        self.CONVERTER.convert(source_dataset, test_dir)

        save_image(
            self._make_image_path(test_dir, "train", "1.jpg"), np.ones((10, 15, 3))
        )  # put the image for dataset
        parsed_dataset = Dataset.import_from(test_dir, "yolo")

        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_dataset_with_exact_image_info(self, helper_tc, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(path="1.jpg", size=(10, 15)),
                    annotations=[
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        self.CONVERTER.convert(source_dataset, test_dir)
        parsed_dataset = Dataset.import_from(test_dir, "yolo", image_info={"1": (10, 15)})
        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self, helper_tc, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="кириллица с пробелом",
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, "yolo")
        compare_datasets(helper_tc, source_dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("save_media", [True, False])
    def test_relative_paths(self, helper_tc, save_media, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", subset="train", media=Image(data=np.ones((4, 2, 3)))),
                DatasetItem(id="subdir1/1", subset="train", media=Image(data=np.ones((2, 6, 3)))),
                DatasetItem(id="subdir2/1", subset="train", media=Image(data=np.ones((5, 4, 3)))),
            ],
            categories=[],
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=save_media)
        parsed_dataset = Dataset.import_from(test_dir, "yolo")
        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self, helper_tc, test_dir):
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

        self.CONVERTER.convert(dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, "yolo")
        compare_datasets(helper_tc, dataset, parsed_dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self, helper_tc, test_dir):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image(data=np.ones((3, 2, 3)))),
            ],
            categories=[],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image(path="2.jpg", size=(3, 2))),
                DatasetItem(3, subset="valid", media=Image(data=np.ones((2, 2, 3)))),
            ],
            categories=[],
        )
        dataset.export(test_dir, "yolo", save_media=True)

        dataset.put(DatasetItem(2, subset="train", media=Image(data=np.ones((3, 2, 3)))))
        dataset.remove(3, "valid")
        dataset.save(save_media=True)

        assert set(os.listdir(osp.join(test_dir, "obj_train_data"))) == {
            "1.txt",
            "2.txt",
            "1.jpg",
            "2.jpg",
        }
        assert set(os.listdir(osp.join(test_dir, "obj_valid_data"))) == set()
        compare_datasets(
            helper_tc,
            expected,
            Dataset.import_from(test_dir, "yolo"),
            require_media=True,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_meta_file(self, helper_tc, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                    ],
                ),
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
        parsed_dataset = Dataset.import_from(test_dir, "yolo")

        assert osp.isfile(osp.join(test_dir, "dataset_meta.json"))
        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_565)
    def test_can_save_and_load_with_custom_subset_name(self, helper_tc, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="anything",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_annotation(),
                        self._generate_random_annotation(),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, "yolo")

        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_565)
    @pytest.mark.parametrize("subset", ["backup", "classes"])
    def test_cant_save_with_reserved_subset_name(self, test_dir, subset):
        self._check_cant_save_with_reserved_subset_name(test_dir, subset)

    def _check_cant_save_with_reserved_subset_name(self, test_dir, subset):
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

        with pytest.raises(DatasetExportError, match=f"Can't export '{subset}' subset"):
            self.CONVERTER.convert(dataset, test_dir)

    @mark_requirement(Requirements.DATUM_609)
    def test_can_save_and_load_without_path_prefix(self, helper_tc, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_annotation(n_of_labels=2),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False)
        parsed_dataset = Dataset.import_from(test_dir, "yolo")

        with open(osp.join(test_dir, "obj.data"), "r") as f:
            lines = f.readlines()
            assert "valid = valid.txt\n" in lines

        with open(osp.join(test_dir, "valid.txt"), "r") as f:
            lines = f.readlines()
            assert "obj_valid_data/3.jpg\n" in lines

        compare_datasets(helper_tc, source_dataset, parsed_dataset)


class Yolo8ConverterTest(YoloConverterTest):
    CONVERTER = Yolo8Converter

    @staticmethod
    def _make_image_path(test_dir: str, subset_name: str, image_id: str):
        return osp.join(test_dir, "images", subset_name, image_id)

    @mark_requirement(Requirements.DATUM_565)
    @pytest.mark.parametrize("subset", ["backup", "classes", "path", "names"])
    def test_cant_save_with_reserved_subset_name(self, test_dir, subset):
        self._check_cant_save_with_reserved_subset_name(test_dir, subset)

    @mark_requirement(Requirements.DATUM_609)
    def test_can_save_and_load_without_path_prefix(self, helper_tc, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_annotation(n_of_labels=2),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False)
        parsed_dataset = Dataset.import_from(test_dir, "yolo")

        with open(osp.join(test_dir, "data.yaml"), "r") as f:
            config = yaml.safe_load(f)
            assert config.get("valid") == "valid.txt"

        with open(osp.join(test_dir, "valid.txt"), "r") as f:
            lines = f.readlines()
            assert "images/valid/3.jpg\n" in lines

        compare_datasets(helper_tc, source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data(self, helper_tc, test_dir):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image(data=np.ones((3, 2, 3)))),
            ],
            categories=[],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(1, subset="train", media=Image(data=np.ones((2, 4, 3)))),
                DatasetItem(2, subset="train", media=Image(path="2.jpg", size=(3, 2))),
                DatasetItem(3, subset="valid", media=Image(data=np.ones((2, 2, 3)))),
            ],
            categories=[],
        )
        dataset.export(test_dir, "yolo8", save_media=True)

        dataset.put(DatasetItem(2, subset="train", media=Image(data=np.ones((3, 2, 3)))))
        dataset.remove(3, "valid")
        dataset.save(save_media=True)

        assert set(os.listdir(osp.join(test_dir, "images", "train"))) == {
            "1.jpg",
            "2.jpg",
        }
        assert set(os.listdir(osp.join(test_dir, "labels", "train"))) == {
            "1.txt",
            "2.txt",
        }
        assert set(os.listdir(osp.join(test_dir, "images", "valid"))) == set()
        assert set(os.listdir(osp.join(test_dir, "labels", "valid"))) == set()
        compare_datasets(
            helper_tc,
            expected,
            Dataset.import_from(test_dir, "yolo"),
            require_media=True,
        )


class Yolo8SegmentationConverterTest(Yolo8ConverterTest):
    CONVERTER = Yolo8SegmentationConverter

    def _generate_random_annotation(self, n_of_labels=10):
        return Polygon(
            points=[random.randint(0, 6) for _ in range(random.randint(3, 7) * 2)],
            label=random.randint(0, n_of_labels - 1),
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_bbox_but_load_polygon(self, helper_tc, test_dir):
        bbox = Bbox(1, 2, 3, 4, label=1)
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Polygon([1, 2, 3, 4, 5, 6, 7, 8, 4, 6], label=0),
                        bbox,
                    ],
                ),
            ],
            categories=["a", "b"],
        )
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Polygon([1, 2, 3, 4, 5, 6, 7, 8, 4, 6], label=0),
                        Polygon(bbox.as_polygon(), label=1),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False)
        parsed_dataset = Dataset.import_from(test_dir, "yolo")
        compare_datasets(helper_tc, expected_dataset, parsed_dataset)


class YoloImporterTest:
    @pytest.mark.parametrize(
        "dataset_dir",
        [
            get_test_asset_path("yolo_dataset", child)
            for child in os.listdir(get_test_asset_path("yolo_dataset"))
        ],
    )
    def test_can_detect(self, dataset_dir: str):
        detected_formats = Environment().detect_dataset(dataset_dir)
        assert detected_formats == [YoloImporter.NAME]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize(
        "dataset_dir",
        [
            get_test_asset_path("yolo_dataset", "yolo"),
            get_test_asset_path("yolo_dataset", "yolo8"),
            get_test_asset_path("yolo_dataset", "yolo8_with_list_of_imgs"),
            get_test_asset_path("yolo_dataset", "yolo8_with_subset_txt"),
            get_test_asset_path("yolo_dataset", "yolo8_with_list_of_names"),
        ],
    )
    def test_can_import(self, helper_tc, dataset_dir):
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

        dataset = Dataset.import_from(dataset_dir, "yolo")

        compare_datasets(helper_tc, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_segmentation(self, helper_tc):
        dataset_dir = get_test_asset_path("yolo_dataset", "yolo8_segmentation")
        expected_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((10, 15, 3))),
                    annotations=[
                        Polygon([1.5, 1.0, 6.0, 1.0, 6.0, 5.0], label=2),
                        Polygon([3.0, 1.5, 6.0, 1.5, 6.0, 7.5, 4.5, 7.5, 3.75, 3.0], label=4),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )

        dataset = Dataset.import_from(dataset_dir, "yolo")

        compare_datasets(helper_tc, expected_dataset, dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_with_exif_rotated_images(self, helper_tc, test_dir):
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

        dataset_path = osp.join(test_dir, "dataset")
        shutil.copytree(get_test_asset_path("yolo_dataset", "yolo"), dataset_path)

        # Add exif rotation for image
        image_path = osp.join(dataset_path, "obj_train_data", "1.jpg")
        img = PILImage.open(image_path)
        exif = img.getexif()
        exif.update([(296, 3), (282, 28.0), (531, 1), (274, 6), (283, 28.0)])
        img.save(image_path, exif=exif)

        dataset = Dataset.import_from(dataset_path, "yolo")

        compare_datasets(helper_tc, expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_673)
    @pytest.mark.parametrize(
        "dataset_dir",
        [
            get_test_asset_path("yolo_dataset", child)
            for child in os.listdir(get_test_asset_path("yolo_dataset"))
        ],
    )
    def test_can_pickle(self, helper_tc, dataset_dir):
        source = Dataset.import_from(dataset_dir, format="yolo")

        parsed = pickle.loads(pickle.dumps(source))  # nosec

        compare_datasets_strict(helper_tc, source, parsed)


class YoloExtractorTest:
    def _prepare_dataset(self, path: str, export_format: str) -> Dataset:
        if export_format == "yolo8_segmentation":
            anno = Polygon(points=[1, 1, 2, 4, 4, 2, 8, 8], label=0)
        else:
            anno = Bbox(1, 1, 2, 4, label=0)
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "a",
                    subset="train",
                    media=Image(np.ones((5, 10, 3))),
                    annotations=[anno],
                )
            ],
            categories=["test"],
        )
        dataset.export(path, export_format, save_media=True)
        return dataset

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @pytest.mark.parametrize("export_format", ["yolo", "yolo8", "yolo8_segmentation"])
    def test_can_parse(self, helper_tc, export_format, test_dir):
        expected = self._prepare_dataset(test_dir, export_format)
        actual = Dataset.import_from(test_dir, "yolo")
        compare_datasets(helper_tc, expected, actual)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize("extractor", [YoloExtractor, Yolo8Extractor])
    def test_can_report_invalid_data_file(self, extractor, test_dir):
        with pytest.raises(DatasetImportError, match="Can't read dataset descriptor file"):
            extractor(test_dir)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "export_format, anno_dir",
        [
            ("yolo", "obj_train_data"),
            ("yolo8", osp.join("labels", "train")),
            ("yolo8_segmentation", osp.join("labels", "train")),
        ],
    )
    def test_can_report_invalid_ann_line_format(self, export_format, anno_dir, test_dir):
        self._prepare_dataset(test_dir, export_format)
        with open(osp.join(test_dir, anno_dir, "a.txt"), "w") as f:
            f.write("1 2 3\n")

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, "yolo").init_cache()
        assert isinstance(capture.value.__cause__, InvalidAnnotationError)
        assert "Unexpected field count" in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "export_format, anno_dir, line",
        [
            ("yolo", "obj_train_data", "10 0.5 0.5 0.5 0.5"),
            ("yolo8", osp.join("labels", "train"), "10 0.5 0.5 0.5 0.5"),
            (
                "yolo8_segmentation",
                osp.join("labels", "train"),
                "10 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5",
            ),
        ],
    )
    def test_can_report_invalid_label(self, export_format, anno_dir, test_dir, line):
        self._prepare_dataset(test_dir, export_format)
        with open(osp.join(test_dir, anno_dir, "a.txt"), "w") as f:
            f.write(f"{line}\n")

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
    @pytest.mark.parametrize(
        "export_format, anno_dir",
        [
            ("yolo", "obj_train_data"),
            ("yolo8", osp.join("labels", "train")),
        ],
    )
    def test_can_report_invalid_field_type(
        self, field, field_name, export_format, anno_dir, test_dir
    ):
        self._prepare_dataset(test_dir, export_format)
        with open(osp.join(test_dir, anno_dir, "a.txt"), "w") as f:
            values = [0, 0.5, 0.5, 0.5, 0.5]
            values[field] = "a"
            f.write(" ".join(str(v) for v in values))

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, "yolo").init_cache()
        assert isinstance(capture.value.__cause__, InvalidAnnotationError)
        assert field_name in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "field, field_name",
        [
            (1, "polygon point 0 x"),
            (2, "polygon point 0 y"),
            (3, "polygon point 1 x"),
            (4, "polygon point 1 y"),
            (5, "polygon point 2 x"),
            (6, "polygon point 2 y"),
        ],
    )
    def test_can_report_invalid_field_type_segmentation(
        self, field: int, field_name: str, test_dir
    ):
        self._prepare_dataset(test_dir, "yolo8_segmentation")
        with open(osp.join(test_dir, "labels", "train", "a.txt"), "w") as f:
            values = [0] + [0.5, 0.5] * 3
            values[field] = "a"
            f.write(" ".join(str(v) for v in values))

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, "yolo").init_cache()
        assert isinstance(capture.value.__cause__, InvalidAnnotationError)
        assert field_name in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "export_format, anno_dir",
        [
            ("yolo", "obj_train_data"),
            ("yolo8", osp.join("labels", "train")),
            ("yolo8_segmentation", osp.join("labels", "train")),
        ],
    )
    def test_can_report_missing_ann_file(self, export_format, anno_dir, test_dir):
        self._prepare_dataset(test_dir, export_format)
        os.remove(osp.join(test_dir, anno_dir, "a.txt"))

        with pytest.raises(ItemImportError) as capture:
            Dataset.import_from(test_dir, "yolo").init_cache()
        assert isinstance(capture.value.__cause__, FileNotFoundError)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "export_format, img_dir",
        [
            ("yolo", "obj_train_data"),
            ("yolo8", osp.join("images", "train")),
            ("yolo8_segmentation", osp.join("images", "train")),
        ],
    )
    def test_can_report_missing_image_info(self, export_format, img_dir, test_dir):
        self._prepare_dataset(test_dir, export_format)
        os.remove(osp.join(test_dir, img_dir, "a.jpg"))

        with pytest.raises(ItemImportError) as capture:
            Dataset.import_from(test_dir, "yolo").init_cache()
        assert isinstance(capture.value.__cause__, DatasetImportError)
        assert "Can't find image info" in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize("export_format", ["yolo", "yolo8"])
    def test_can_report_missing_subset_info(self, export_format, test_dir):
        self._prepare_dataset(test_dir, export_format)
        os.remove(osp.join(test_dir, "train.txt"))

        with pytest.raises(InvalidAnnotationError, match="subset list file"):
            Dataset.import_from(test_dir, "yolo").init_cache()

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_subset_folder(self, test_dir):
        dataset_path = osp.join(test_dir, "dataset")
        shutil.copytree(get_test_asset_path("yolo_dataset", "yolo8"), dataset_path)
        shutil.rmtree(osp.join(dataset_path, "images", "train"))

        with pytest.raises(InvalidAnnotationError, match="subset image folder"):
            Dataset.import_from(dataset_path, "yolo").init_cache()
