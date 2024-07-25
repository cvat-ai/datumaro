import copy
import os
import os.path as osp
import pickle  # nosec - disable B403:import_pickle check
import random
import shutil

import numpy as np
import pytest
import yaml
from PIL import ExifTags
from PIL import Image as PILImage

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    LabelCategories,
    Points,
    PointsCategories,
    Polygon,
    Skeleton,
)
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
    Yolo8ObbConverter,
    Yolo8PoseConverter,
    Yolo8SegmentationConverter,
    YoloConverter,
)
from datumaro.plugins.yolo_format.extractor import (
    Yolo8Extractor,
    Yolo8ObbExtractor,
    Yolo8PoseExtractor,
    Yolo8SegmentationExtractor,
    YoloExtractor,
)
from datumaro.plugins.yolo_format.importer import (
    Yolo8Importer,
    Yolo8ObbImporter,
    Yolo8PoseImporter,
    Yolo8SegmentationImporter,
    YoloImporter,
)
from datumaro.util.image import save_image
from datumaro.util.test_utils import compare_datasets, compare_datasets_strict

from ...requirements import Requirements, mark_requirement
from ...utils.assets import get_test_asset_path


class CompareDatasetMixin:
    @pytest.fixture(autouse=True)
    def setup(self, helper_tc):
        self.helper_tc = helper_tc

    def compare_datasets(self, *args, **kwargs):
        compare_datasets(self.helper_tc, *args, **kwargs)


class CompareDatasetsRotationMixin(CompareDatasetMixin):
    def compare_datasets(self, expected, actual, **kwargs):
        actual_copy = copy.deepcopy(actual)
        compare_datasets(self.helper_tc, expected, actual, ignored_attrs=["rotation"], **kwargs)
        for item_a, item_b in zip(expected, actual_copy):
            for ann_a, ann_b in zip(item_a.annotations, item_b.annotations):
                assert ("rotation" in ann_a.attributes) == ("rotation" in ann_b.attributes)
                assert (
                    abs(ann_a.attributes.get("rotation", 0) - ann_b.attributes.get("rotation", 0))
                    < 0.01
                )


class YoloConverterTest(CompareDatasetMixin):
    CONVERTER = YoloConverter
    IMPORTER = YoloImporter

    def _generate_random_bbox(self, n_of_labels=10, **kwargs):
        return Bbox(
            x=random.randint(0, 4),
            y=random.randint(0, 4),
            w=random.randint(1, 4),
            h=random.randint(1, 4),
            label=random.randint(0, n_of_labels - 1),
            attributes=kwargs,
        )

    def _generate_random_annotation(self, n_of_labels=10):
        return self._generate_random_bbox(n_of_labels=n_of_labels)

    @staticmethod
    def _make_image_path(test_dir: str, subset_name: str, image_id: str):
        return osp.join(test_dir, f"obj_{subset_name}_data", image_id)

    def _generate_random_dataset(self, recipes, n_of_labels=10):
        items = [
            DatasetItem(
                id=recipe.get("id", index + 1),
                subset=recipe.get("subset", "train"),
                media=recipe.get(
                    "media",
                    Image(data=np.ones((random.randint(8, 10), random.randint(8, 10), 3))),
                ),
                annotations=[
                    self._generate_random_annotation(n_of_labels=n_of_labels)
                    for _ in range(recipe.get("annotations", 1))
                ],
            )
            for index, recipe in enumerate(recipes)
        ]
        return Dataset.from_iterable(
            items,
            categories=["label_" + str(i) for i in range(n_of_labels)],
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self, helper_tc, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {"annotations": 2},
                {"annotations": 3},
                {"annotations": 4},
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_image_info(self, helper_tc, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {
                    "annotations": 2,
                    "media": Image(path="1.jpg", size=(10, 15)),
                },
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir)

        save_image(
            self._make_image_path(test_dir, "train", "1.jpg"), np.ones((10, 15, 3))
        )  # put the image for dataset
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_load_dataset_with_exact_image_info(self, helper_tc, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {
                    "annotations": 2,
                    "media": Image(path="1.jpg", size=(10, 15)),
                },
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir)
        parsed_dataset = Dataset.import_from(
            test_dir, self.IMPORTER.NAME, image_info={"1": (10, 15)}
        )
        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self, helper_tc, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {
                    "id": "кириллица с пробелом",
                    "annotations": 2,
                },
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(source_dataset, parsed_dataset, require_media=True)

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
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(source_dataset, parsed_dataset)

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
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(dataset, parsed_dataset, require_media=True)

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
        self.compare_datasets(
            expected,
            Dataset.import_from(test_dir, "yolo"),
            require_media=True,
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_meta_file(self, helper_tc, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {"annotations": 2},
                {"annotations": 3},
                {"annotations": 4},
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        assert osp.isfile(osp.join(test_dir, "dataset_meta.json"))
        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_565)
    def test_can_save_and_load_with_custom_subset_name(self, helper_tc, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {"annotations": 2, "subset": "anything", "id": 3},
            ]
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        self.compare_datasets(source_dataset, parsed_dataset)

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
        source_dataset = self._generate_random_dataset(
            [
                {"subset": "valid", "id": 3},
            ],
            n_of_labels=2,
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        with open(osp.join(test_dir, "obj.data"), "r") as f:
            lines = f.readlines()
            assert "valid = valid.txt\n" in lines

        with open(osp.join(test_dir, "valid.txt"), "r") as f:
            lines = f.readlines()
            assert "obj_valid_data/3.jpg\n" in lines

        self.compare_datasets(source_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_export_rotated_bbox(self, test_dir):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_bbox(n_of_labels=2, rotation=30.0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )
        with pytest.raises(DatasetExportError) as capture:
            dataset.export(test_dir, self.CONVERTER.NAME)
        assert isinstance(capture.value.__cause__, DatasetExportError)
        assert "Can't export rotated bbox" in str(capture.value.__cause__)


class Yolo8ConverterTest(YoloConverterTest):
    CONVERTER = Yolo8Converter
    IMPORTER = Yolo8Importer

    @staticmethod
    def _make_image_path(test_dir: str, subset_name: str, image_id: str):
        return osp.join(test_dir, "images", subset_name, image_id)

    @mark_requirement(Requirements.DATUM_565)
    @pytest.mark.parametrize("subset", ["backup", "classes", "path", "names"])
    def test_cant_save_with_reserved_subset_name(self, test_dir, subset):
        self._check_cant_save_with_reserved_subset_name(test_dir, subset)

    @mark_requirement(Requirements.DATUM_609)
    def test_can_save_and_load_without_path_prefix(self, helper_tc, test_dir):
        source_dataset = self._generate_random_dataset(
            [
                {"subset": "valid", "id": 3},
            ],
            n_of_labels=2,
        )

        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, add_path_prefix=False)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)

        with open(osp.join(test_dir, "data.yaml"), "r") as f:
            config = yaml.safe_load(f)
            assert config.get("valid") == "valid.txt"

        with open(osp.join(test_dir, "valid.txt"), "r") as f:
            lines = f.readlines()
            assert "images/valid/3.jpg\n" in lines

        self.compare_datasets(source_dataset, parsed_dataset)

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
        dataset.export(test_dir, self.CONVERTER.NAME, save_media=True)

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
        self.compare_datasets(
            expected,
            Dataset.import_from(test_dir, self.IMPORTER.NAME),
            require_media=True,
        )


class Yolo8SegmentationConverterTest(Yolo8ConverterTest):
    CONVERTER = Yolo8SegmentationConverter
    IMPORTER = Yolo8SegmentationImporter

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
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(expected_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_export_rotated_bbox(self, test_dir, helper_tc):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        Bbox(
                            x=1,
                            y=2,
                            w=3,
                            h=4,
                            label=0,
                            attributes=dict(rotation=30.0),
                        )
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
                        Polygon(
                            points=[2.2, 1.52, 4.8, 3.02, 2.8, 6.48, 0.2, 4.98],
                            label=0,
                        )
                    ],
                ),
            ],
            categories=["a", "b"],
        )
        dataset.export(test_dir, self.CONVERTER.NAME, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(expected_dataset, parsed_dataset)


class Yolo8ObbConverterTest(CompareDatasetsRotationMixin, Yolo8ConverterTest):
    CONVERTER = Yolo8ObbConverter
    IMPORTER = Yolo8ObbImporter

    def _generate_random_annotation(self, n_of_labels=10):
        return self._generate_random_bbox(n_of_labels=n_of_labels, rotation=random.randint(10, 350))

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_export_rotated_bbox(self, test_dir):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=3,
                    subset="valid",
                    media=Image(data=np.ones((8, 8, 3))),
                    annotations=[
                        self._generate_random_bbox(n_of_labels=2, rotation=30.0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )
        source_dataset.export(test_dir, self.CONVERTER.NAME, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        assert abs(list(parsed_dataset)[0].annotations[0].attributes["rotation"] - 30) < 0.001
        self.compare_datasets(source_dataset, parsed_dataset)


class Yolo8PoseConverterTest(Yolo8ConverterTest):
    CONVERTER = Yolo8PoseConverter
    IMPORTER = Yolo8PoseImporter

    def _generate_random_skeleton_annotation(self, skeleton_label_to_point_labels, n_of_labels=10):
        label_id = random.choice(list(skeleton_label_to_point_labels.keys()))
        return Skeleton(
            [
                Points(
                    [random.randint(1, 7), random.randint(1, 7)],
                    [random.randint(0, 2)],
                    label=label,
                )
                for label in skeleton_label_to_point_labels[label_id]
            ],
            label=label_id,
        )

    def _generate_random_dataset(self, recipes, n_of_labels=10):
        n_of_points_in_skeleton = random.randint(3, 8)
        labels = [f"skeleton_label_{index}" for index in range(n_of_labels)] + [
            (f"skeleton_label_{parent_index}_point_{point_index}", f"skeleton_label_{parent_index}")
            for parent_index in range(n_of_labels)
            for point_index in range(n_of_points_in_skeleton)
        ]
        skeleton_label_to_point_labels = {
            skeleton_label_id: [
                label_id
                for label_id, label in enumerate(labels)
                if isinstance(label, tuple) and label[1] == f"skeleton_label_{skeleton_label_id}"
            ]
            for skeleton_label_id, skeleton_label in enumerate(labels)
            if isinstance(skeleton_label, str)
        }
        items = [
            DatasetItem(
                id=recipe.get("id", index + 1),
                subset=recipe.get("subset", "train"),
                media=recipe.get(
                    "media",
                    Image(data=np.ones((random.randint(8, 10), random.randint(8, 10), 3))),
                ),
                annotations=[
                    self._generate_random_skeleton_annotation(
                        skeleton_label_to_point_labels,
                        n_of_labels=n_of_labels,
                    )
                    for _ in range(recipe.get("annotations", 1))
                ],
            )
            for index, recipe in enumerate(recipes)
        ]

        point_categories = PointsCategories.from_iterable(
            [
                (
                    index,
                    [
                        f"skeleton_label_{index}_point_{point_index}"
                        for point_index in range(n_of_points_in_skeleton)
                    ],
                    set(),
                )
                for index in range(n_of_labels)
            ]
        )

        return Dataset.from_iterable(
            items,
            categories={
                AnnotationType.label: LabelCategories.from_iterable(labels),
                AnnotationType.points: point_categories,
            },
        )

    def test_export_rotated_bbox(self):
        pass

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_loses_skeleton_edges_and_point_labels_on_save_load_without_meta_file(self, test_dir):
        items = [
            DatasetItem(
                id="1",
                subset="train",
                media=Image(data=np.ones((5, 10, 3))),
                annotations=[
                    Skeleton(
                        [
                            Points([1.5, 2.0], [2], label=1),
                            Points([4.5, 4.0], [2], label=2),
                        ],
                        label=0,
                    ),
                ],
            ),
        ]
        source_dataset = Dataset.from_iterable(
            items,
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "skeleton_label",
                        ("point_label_1", "skeleton_label"),
                        ("point_label_2", "skeleton_label"),
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(0, ["point_label_1", "point_label_2"], {(0, 1)})],
                ),
            },
        )
        expected_dataset = Dataset.from_iterable(
            items,
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "skeleton_label",
                        ("skeleton_label_point_0", "skeleton_label"),
                        ("skeleton_label_point_1", "skeleton_label"),
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(0, ["skeleton_label_point_0", "skeleton_label_point_1"], set())],
                ),
            },
        )
        self.CONVERTER.convert(source_dataset, test_dir, save_media=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        self.compare_datasets(expected_dataset, parsed_dataset)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_meta_file(self, helper_tc, test_dir):
        items = [
            DatasetItem(
                id="1",
                subset="train",
                media=Image(data=np.ones((5, 10, 3))),
                annotations=[
                    Skeleton(
                        [
                            Points([1.5, 2.0], [2], label=1),
                            Points([4.5, 4.0], [2], label=2),
                        ],
                        label=0,
                    ),
                ],
            ),
        ]
        source_dataset = Dataset.from_iterable(
            items,
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "skeleton_label",
                        ("point_label_1", "skeleton_label"),
                        ("point_label_2", "skeleton_label"),
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(0, ["point_label_1", "point_label_2"], {(0, 1)})],
                ),
            },
        )
        self.CONVERTER.convert(source_dataset, test_dir, save_media=True, save_dataset_meta=True)
        parsed_dataset = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        assert osp.isfile(osp.join(test_dir, "dataset_meta.json"))
        self.compare_datasets(source_dataset, parsed_dataset)


class YoloImporterTest(CompareDatasetMixin):
    IMPORTER = YoloImporter
    ASSETS = ["yolo"]

    def test_can_detect(self):
        dataset_dir = get_test_asset_path("yolo_dataset", "yolo")
        detected_formats = Environment().detect_dataset(dataset_dir)
        assert detected_formats == [self.IMPORTER.NAME]

    @staticmethod
    def _asset_dataset():
        return Dataset.from_iterable(
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

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import(self, helper_tc):
        expected_dataset = self._asset_dataset()
        for asset in self.ASSETS:
            dataset_dir = get_test_asset_path("yolo_dataset", asset)
            dataset = Dataset.import_from(dataset_dir, self.IMPORTER.NAME)
            self.compare_datasets(expected_dataset, dataset)

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
        exif.update(
            [
                (ExifTags.Base.ResolutionUnit, 3),
                (ExifTags.Base.XResolution, 28.0),
                (ExifTags.Base.YCbCrPositioning, 1),
                (ExifTags.Base.Orientation, 6),
                (ExifTags.Base.YResolution, 28.0),
            ]
        )
        img.save(image_path, exif=exif)

        dataset = Dataset.import_from(dataset_path, "yolo")

        self.compare_datasets(expected_dataset, dataset, require_media=True)

    @mark_requirement(Requirements.DATUM_673)
    def test_can_pickle(self, helper_tc):
        for asset in self.ASSETS:
            dataset_dir = get_test_asset_path("yolo_dataset", asset)
            source = Dataset.import_from(dataset_dir, format=self.IMPORTER.NAME)
            parsed = pickle.loads(pickle.dumps(source))  # nosec
            compare_datasets_strict(helper_tc, source, parsed)


class Yolo8ImporterTest(YoloImporterTest):
    IMPORTER = Yolo8Importer
    ASSETS = [
        "yolo8",
        "yolo8_with_list_of_imgs",
        "yolo8_with_subset_txt",
        "yolo8_with_list_of_names",
    ]

    def test_can_detect(self):
        for asset in self.ASSETS:
            dataset_dir = get_test_asset_path("yolo_dataset", asset)
            detected_formats = Environment().detect_dataset(dataset_dir)
            assert set(detected_formats) == {
                Yolo8Importer.NAME,
                Yolo8SegmentationImporter.NAME,
                Yolo8ObbImporter.NAME,
            }


class Yolo8SegmentationImporterTest(Yolo8ImporterTest):
    IMPORTER = Yolo8SegmentationImporter
    ASSETS = [
        "yolo8_segmentation",
    ]

    @staticmethod
    def _asset_dataset():
        return Dataset.from_iterable(
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


class Yolo8ObbImporterTest(CompareDatasetsRotationMixin, Yolo8ImporterTest):
    IMPORTER = Yolo8ObbImporter
    ASSETS = ["yolo8_obb"]

    @staticmethod
    def _asset_dataset():
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=2, attributes=dict(rotation=30)),
                        Bbox(3, 2, 6, 2, label=4, attributes=dict(rotation=120)),
                    ],
                ),
            ],
            categories=["label_" + str(i) for i in range(10)],
        )


class Yolo8PoseImporterTest(Yolo8ImporterTest):
    IMPORTER = Yolo8PoseImporter
    ASSETS = [
        "yolo8_pose",
        "yolo8_pose_two_values_per_point",
    ]

    def test_can_detect(self):
        for asset in self.ASSETS:
            dataset_dir = get_test_asset_path("yolo_dataset", asset)
            detected_formats = Environment().detect_dataset(dataset_dir)
            assert detected_formats == [self.IMPORTER.NAME]

    @staticmethod
    def _asset_dataset():
        return Dataset.from_iterable(
            [
                DatasetItem(
                    id="1",
                    subset="train",
                    media=Image(data=np.ones((5, 10, 3))),
                    annotations=[
                        Skeleton(
                            [
                                Points([1.5, 2.0], [2], label=1),
                                Points([4.5, 4.0], [2], label=2),
                                Points([7.5, 6.0], [2], label=3),
                            ],
                            label=0,
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "skeleton_label",
                        ("skeleton_label_point_0", "skeleton_label"),
                        ("skeleton_label_point_1", "skeleton_label"),
                        ("skeleton_label_point_2", "skeleton_label"),
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (
                            0,
                            [
                                "skeleton_label_point_0",
                                "skeleton_label_point_1",
                                "skeleton_label_point_2",
                            ],
                            set(),
                        )
                    ],
                ),
            },
        )


class YoloExtractorTest:
    IMPORTER = YoloImporter
    EXTRACTOR = YoloExtractor

    def _prepare_dataset(self, path: str, anno=None) -> Dataset:
        if anno is None:
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
        dataset.export(path, self.EXTRACTOR.NAME, save_media=True)
        return dataset

    @staticmethod
    def _get_annotation_dir(subset="train"):
        return f"obj_{subset}_data"

    @staticmethod
    def _get_image_dir(subset="train"):
        return f"obj_{subset}_data"

    @staticmethod
    def _make_some_annotation_values():
        return [0.5, 0.5, 0.5, 0.5]

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_parse(self, helper_tc, test_dir):
        expected = self._prepare_dataset(test_dir)
        actual = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        compare_datasets(helper_tc, expected, actual)

    def test_can_report_invalid_data_file(self, test_dir):
        with pytest.raises(DatasetImportError, match="Can't read dataset descriptor file"):
            self.EXTRACTOR(test_dir)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_ann_line_format(self, test_dir):
        self._prepare_dataset(test_dir)
        with open(osp.join(test_dir, self._get_annotation_dir(), "a.txt"), "w") as f:
            f.write("1 2 3\n")

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
        assert isinstance(capture.value.__cause__, InvalidAnnotationError)
        assert "Unexpected field count" in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_invalid_label(self, test_dir):
        self._prepare_dataset(test_dir)
        with open(osp.join(test_dir, self._get_annotation_dir(), "a.txt"), "w") as f:
            f.write(" ".join(str(v) for v in [10] + self._make_some_annotation_values()))

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
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
    def test_can_report_invalid_field_type(self, field, field_name, test_dir):
        self._check_can_report_invalid_field_type(field, field_name, test_dir)

    def _check_can_report_invalid_field_type(self, field, field_name, test_dir):
        self._prepare_dataset(test_dir)
        with open(osp.join(test_dir, self._get_annotation_dir(), "a.txt"), "w") as f:
            values = [0] + self._make_some_annotation_values()
            values[field] = "a"
            f.write(" ".join(str(v) for v in values))

        with pytest.raises(AnnotationImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
        assert isinstance(capture.value.__cause__, InvalidAnnotationError)
        assert field_name in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_ann_file(self, test_dir):
        self._prepare_dataset(test_dir)
        os.remove(osp.join(test_dir, self._get_annotation_dir(), "a.txt"))

        with pytest.raises(ItemImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
        assert isinstance(capture.value.__cause__, FileNotFoundError)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_image_info(self, test_dir):
        self._prepare_dataset(test_dir)
        os.remove(osp.join(test_dir, self._get_image_dir(), "a.jpg"))

        with pytest.raises(ItemImportError) as capture:
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()
        assert isinstance(capture.value.__cause__, DatasetImportError)
        assert "Can't find image info" in str(capture.value.__cause__)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_subset_info(self, test_dir):
        self._prepare_dataset(test_dir)
        os.remove(osp.join(test_dir, "train.txt"))

        with pytest.raises(InvalidAnnotationError, match="subset list file"):
            Dataset.import_from(test_dir, self.IMPORTER.NAME).init_cache()


class Yolo8ExtractorTest(YoloExtractorTest):
    IMPORTER = Yolo8Importer
    EXTRACTOR = Yolo8Extractor

    @staticmethod
    def _get_annotation_dir(subset="train"):
        return osp.join("labels", subset)

    @staticmethod
    def _get_image_dir(subset="train"):
        return osp.join("images", subset)

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    def test_can_report_missing_subset_folder(self, test_dir):
        dataset_path = osp.join(test_dir, "dataset")
        shutil.copytree(get_test_asset_path("yolo_dataset", self.IMPORTER.NAME), dataset_path)
        shutil.rmtree(osp.join(dataset_path, "images", "train"))

        with pytest.raises(InvalidAnnotationError, match="subset image folder"):
            Dataset.import_from(dataset_path, self.IMPORTER.NAME).init_cache()


class Yolo8SegmentationExtractorTest(Yolo8ExtractorTest):
    IMPORTER = Yolo8SegmentationImporter
    EXTRACTOR = Yolo8SegmentationExtractor

    def _prepare_dataset(self, path: str, anno=None) -> Dataset:
        return super()._prepare_dataset(
            path, anno=Polygon(points=[1, 1, 2, 4, 4, 2, 8, 8], label=0)
        )

    @staticmethod
    def _make_some_annotation_values():
        return [0.5, 0.5] * 3

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
    def test_can_report_invalid_field_type(self, field, field_name, test_dir):
        self._check_can_report_invalid_field_type(field, field_name, test_dir)


class Yolo8ObbExtractorTest(Yolo8ExtractorTest):
    IMPORTER = Yolo8ObbImporter
    EXTRACTOR = Yolo8ObbExtractor

    def _prepare_dataset(self, path: str, anno=None) -> Dataset:
        return super()._prepare_dataset(
            path, anno=Bbox(1, 1, 2, 4, label=0, attributes=dict(rotation=30))
        )

    @staticmethod
    def _make_some_annotation_values():
        return [0.5, 0.5] * 4

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_parse(self, helper_tc, test_dir):
        expected = self._prepare_dataset(test_dir)
        actual = Dataset.import_from(test_dir, self.IMPORTER.NAME)
        assert abs(list(actual)[0].annotations[0].attributes["rotation"] - 30) < 0.001
        compare_datasets(helper_tc, expected, actual, ignored_attrs=["rotation"])

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "field, field_name",
        [
            (1, "bbox point 0 x"),
            (2, "bbox point 0 y"),
            (3, "bbox point 1 x"),
            (4, "bbox point 1 y"),
            (5, "bbox point 2 x"),
            (6, "bbox point 2 y"),
        ],
    )
    def test_can_report_invalid_field_type(self, field, field_name, test_dir):
        self._check_can_report_invalid_field_type(field, field_name, test_dir)


class Yolo8PoseExtractorTest(Yolo8ExtractorTest):
    IMPORTER = Yolo8PoseImporter
    EXTRACTOR = Yolo8PoseExtractor

    def _prepare_dataset(self, path: str) -> Dataset:
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    "a",
                    subset="train",
                    media=Image(np.ones((5, 10, 3))),
                    annotations=[
                        Skeleton(
                            [
                                Points([1, 2], [Points.Visibility.visible.value], label=1),
                                Points([3, 6], [Points.Visibility.visible.value], label=2),
                                Points([4, 5], [Points.Visibility.visible.value], label=3),
                                Points([8, 7], [Points.Visibility.visible.value], label=4),
                            ],
                            label=0,
                        )
                    ],
                )
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(
                    [
                        "test",
                        ("test_point_0", "test"),
                        ("test_point_1", "test"),
                        ("test_point_2", "test"),
                        ("test_point_3", "test"),
                    ]
                ),
                AnnotationType.points: PointsCategories.from_iterable(
                    [(0, ["test_point_0", "test_point_1", "test_point_2", "test_point_3"], set())]
                ),
            },
        )
        dataset.export(path, self.EXTRACTOR.NAME, save_media=True)
        return dataset

    @staticmethod
    def _make_some_annotation_values():
        return [0.5, 0.5, 0.5, 0.5] + [0.5, 0.5, 2] * 4

    @mark_requirement(Requirements.DATUM_ERROR_REPORTING)
    @pytest.mark.parametrize(
        "field, field_name",
        [
            (5, "skeleton point 0 x"),
            (6, "skeleton point 0 y"),
            (7, "skeleton point 0 visibility"),
            (8, "skeleton point 1 x"),
            (9, "skeleton point 1 y"),
            (10, "skeleton point 1 visibility"),
        ],
    )
    def test_can_report_invalid_field_type(self, field, field_name, test_dir):
        self._check_can_report_invalid_field_type(field, field_name, test_dir)
