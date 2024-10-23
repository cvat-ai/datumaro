import os
import os.path as osp
from functools import partial
from pathlib import Path
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Cuboid3d,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    Skeleton,
)
from datumaro.components.environment import Environment
from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image, PointCloud
from datumaro.components.project import Dataset
from datumaro.plugins.data_formats.datumaro.converter import DatumaroConverter
from datumaro.plugins.data_formats.datumaro.extractor import DatumaroImporter
from datumaro.util import parse_json_file
from datumaro.util.mask_tools import generate_colormap

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import (
    Dimensions,
    TestDir,
    check_save_and_load,
    compare_datasets,
    compare_datasets_strict,
)


class DatumaroConverterTest(TestCase):
    def _test_save_and_load(
        self,
        source_dataset,
        converter,
        test_dir,
        target_dataset=None,
        importer_args=None,
        compare=compare_datasets_strict,
        **kwargs,
    ):
        return check_save_and_load(
            self,
            source_dataset,
            converter,
            test_dir,
            importer="datumaro",
            target_dataset=target_dataset,
            importer_args=importer_args,
            compare=compare,
            **kwargs,
        )

    @property
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_dataset(self):
        label_categories = LabelCategories(attributes={"a", "b", "score"})
        for i in range(5):
            label_categories.add("cat" + str(i), attributes={"x", "y"})

        mask_categories = MaskCategories(generate_colormap(len(label_categories.items)))

        points_categories = PointsCategories()
        for index, _ in enumerate(label_categories.items):
            points_categories.add(index, ["cat1", "cat2"], joints=[[0, 1]])

        return Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Caption("hello", id=1),
                        Caption("world", id=2, group=5),
                        Label(
                            2,
                            id=3,
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            3,
                            4,
                            label=4,
                            id=4,
                            z_order=1,
                            attributes={
                                "score": 1.0,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            7,
                            8,
                            id=5,
                            group=5,
                            attributes={
                                "a": 1.5,
                                "b": "text",
                            },
                        ),
                        Points(
                            [1, 2, 2, 0, 1, 1],
                            label=0,
                            id=5,
                            z_order=4,
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                        Mask(
                            label=3,
                            id=5,
                            z_order=2,
                            image=np.ones((2, 3)),
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                    ],
                ),
                DatasetItem(
                    id=21,
                    subset="train",
                    annotations=[
                        Caption("test"),
                        Label(2),
                        Bbox(1, 2, 3, 4, label=5, id=42, group=42),
                    ],
                ),
                DatasetItem(
                    id=2,
                    subset="val",
                    annotations=[
                        PolyLine([1, 2, 3, 4, 5, 6, 7, 8], id=11, z_order=1),
                        Polygon([1, 2, 3, 4, 5, 6, 7, 8], id=12, z_order=4),
                    ],
                ),
                DatasetItem(
                    id=1,
                    subset="test",
                    annotations=[
                        Cuboid3d(
                            [1.0, 2.0, 3.0],
                            [2.0, 2.0, 4.0],
                            [1.0, 3.0, 4.0],
                            id=6,
                            label=0,
                            attributes={"occluded": True},
                            group=6,
                        )
                    ],
                ),
                DatasetItem(id=42, subset="test", attributes={"a1": 5, "a2": "42"}),
                DatasetItem(id=42),
                DatasetItem(id=43, media=Image(path="1/b/c.qq", size=(2, 4))),
            ],
            categories={
                AnnotationType.label: label_categories,
                AnnotationType.mask: mask_categories,
                AnnotationType.points: points_categories,
            },
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load(self):
        with TestDir() as test_dir:
            self._test_save_and_load(
                self.test_dataset, partial(DatumaroConverter.convert, save_media=True), test_dir
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_no_save_media(self):
        with TestDir() as test_dir:
            self._test_save_and_load(
                self.test_dataset,
                partial(DatumaroConverter.convert, save_media=True),
                test_dir,
                compare=None,
                require_media=False,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_pcd_dataset(self):
        dataset_path = get_test_asset_path("datumaro_dataset", "with_pcd")

        label_categories = LabelCategories(attributes={"occluded"})
        label_categories.add("cat")

        points_categories = PointsCategories()

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id="0000000000",
                    annotations=[
                        Cuboid3d(
                            [0.84, -0.22, 0.64],
                            [0.0, 0.0, 0.0],
                            [1.0, 1.0, 1.0],
                            label=0,
                            attributes={"occluded": False},
                        )
                    ],
                    media=PointCloud(
                        os.path.join(dataset_path, "point_clouds", "default", "0000000000.pcd")
                    ),
                    attributes={"frame": 0},
                ),
                DatasetItem(
                    id="0000000001",
                    media=PointCloud(
                        os.path.join(dataset_path, "point_clouds", "default", "0000000001.pcd")
                    ),
                    attributes={"frame": 1},
                ),
                DatasetItem(
                    id="0000000002",
                    media=PointCloud(
                        os.path.join(dataset_path, "point_clouds", "default", "0000000002.pcd")
                    ),
                    attributes={"frame": 2},
                ),
            ],
            categories={
                AnnotationType.label: label_categories,
                AnnotationType.points: points_categories,
            },
            media_type=PointCloud,
        )

        compare_datasets_strict(self, expected, Dataset.load(dataset_path))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_import_skeleton_dataset(self):
        dataset_path = get_test_asset_path("datumaro_dataset", "with_skeleton")

        label_categories = LabelCategories.from_iterable(
            [
                ("skeleton-label",),
                ("point1-label", "skeleton-label"),
                ("point3-label", "skeleton-label"),
                ("point2-label", "skeleton-label"),
            ]
        )
        label_categories.attributes = {
            "point3-attribute-checkbox",
            "occluded",
            "point2-attribute-text",
        }
        points_categories = PointsCategories.from_iterable(
            [
                (0, ["point1-label", "point3-label", "point2-label"], {(2, 3), (1, 2)}),
            ]
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id="100",
                    subset="default",
                    media=Image(data=np.ones((10, 6, 3))),
                    annotations=[
                        Skeleton(
                            [
                                Points(
                                    [0.9, 3.53],
                                    label=1,
                                    attributes={},
                                ),
                                Points(
                                    [2.45, 7.6],
                                    label=2,
                                    attributes={"point3-attribute-checkbox": True},
                                ),
                                Points(
                                    [5.2, 2.5],
                                    label=3,
                                    attributes={"point2-attribute-text": "some text"},
                                ),
                            ],
                            label=0,
                            attributes={"occluded": False, "keyframe": False},
                        ),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: label_categories,
                AnnotationType.points: points_categories,
            },
        )
        compare_datasets_strict(self, expected, Dataset.load(dataset_path))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_detect(self):
        with TestDir() as test_dir:
            DatumaroConverter.convert(self.test_dataset, save_dir=test_dir)

            detected_formats = Environment().detect_dataset(test_dir)
            self.assertEqual([DatumaroImporter.NAME], detected_formats)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_relative_paths(self):
        test_dataset = Dataset.from_iterable(
            [
                DatasetItem(id="1", media=Image(data=np.ones((4, 2, 3)))),
                DatasetItem(id="subdir1/1", media=Image(data=np.ones((2, 6, 3)))),
                DatasetItem(id="subdir2/1", media=Image(data=np.ones((5, 4, 3)))),
            ]
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                test_dataset, partial(DatumaroConverter.convert, save_media=True), test_dir
            )

    @mark_requirement(Requirements.DATUM_231)
    def test_can_save_dataset_with_cjk_categories(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="train",
                    media=Image(data=np.ones((4, 4, 3))),
                    annotations=[
                        Bbox(0, 1, 2, 2, label=0, group=1, id=1, attributes={"is_crowd": False}),
                    ],
                    attributes={"id": 1},
                ),
                DatasetItem(
                    id=2,
                    subset="train",
                    media=Image(data=np.ones((4, 4, 3))),
                    annotations=[
                        Bbox(1, 0, 2, 2, label=1, group=2, id=2, attributes={"is_crowd": False}),
                    ],
                    attributes={"id": 2},
                ),
                DatasetItem(
                    id=3,
                    subset="train",
                    media=Image(data=np.ones((4, 4, 3))),
                    annotations=[
                        Bbox(0, 1, 2, 2, label=2, group=3, id=3, attributes={"is_crowd": False}),
                    ],
                    attributes={"id": 3},
                ),
            ],
            categories=["고양이", "ネコ", "猫"],
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected, partial(DatumaroConverter.convert, save_media=True), test_dir
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_dataset_with_cyrillic_and_spaces_in_filename(self):
        test_dataset = Dataset.from_iterable(
            [DatasetItem(id="кириллица с пробелом", media=Image(data=np.ones((4, 2, 3))))]
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                test_dataset, partial(DatumaroConverter.convert, save_media=True), test_dir
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_image_with_arbitrary_extension(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    id="q/1",
                    media=Image(path="q/1.JPEG", data=np.zeros((4, 3, 3))),
                    attributes={"frame": 1},
                ),
                DatasetItem(
                    id="a/b/c/2",
                    media=Image(path="a/b/c/2.bmp", data=np.zeros((3, 4, 3))),
                    attributes={"frame": 2},
                ),
            ]
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                expected, partial(DatumaroConverter.convert, save_media=True), test_dir
            )

    @mark_requirement(Requirements.DATUM_CVAT_AI_BUG_20)
    def test_save_and_load_with_only_specific_media_field(self):
        # item 'media' field should not be present, when any other media field is present
        with TestDir() as test_dir:
            self._test_save_and_load(
                self.test_dataset, partial(DatumaroConverter.convert, save_media=True), test_dir
            )

            for exported_json_path in Path(test_dir).glob("*/**/*.json"):
                exported_json_data = parse_json_file(os.fspath(exported_json_path))

                for item_data in exported_json_data["items"]:
                    self.assertTrue(
                        len([k for k in item_data if k in ["media", "image", "point_cloud"]]) == 0
                        or ("media" in item_data)
                        ^ ("image" in item_data or "point_cloud" in item_data)
                    )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data_with_direct_changes(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(1, subset="a"),
                DatasetItem(2, subset="a", media=Image(data=np.ones((3, 2, 3)))),
                DatasetItem(2, subset="b"),
            ]
        )

        with TestDir() as path:
            # generate initial dataset
            dataset = Dataset.from_iterable(
                [
                    # modified subset
                    DatasetItem(1, subset="a"),
                    # unmodified subset
                    DatasetItem(2, subset="b"),
                    # removed subset
                    DatasetItem(3, subset="c", media=Image(data=np.ones((2, 2, 3)))),
                ]
            )
            dataset.save(path, save_media=True)

            dataset.put(DatasetItem(2, subset="a", media=Image(data=np.ones((3, 2, 3)))))
            dataset.remove(3, "c")
            dataset.save(save_media=True)

            self.assertEqual({"a.json", "b.json"}, set(os.listdir(osp.join(path, "annotations"))))
            self.assertEqual({"2.jpg"}, set(os.listdir(osp.join(path, "images", "a"))))
            compare_datasets_strict(self, expected, Dataset.load(path))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_inplace_save_writes_only_updated_data_with_transforms(self):
        with TestDir() as path:
            expected = Dataset.from_iterable(
                [
                    DatasetItem(2, subset="test"),
                    DatasetItem(3, subset="train", media=Image(data=np.ones((2, 2, 3)))),
                    DatasetItem(4, subset="test", media=Image(data=np.ones((2, 3, 3)))),
                ],
                media_type=Image,
            )
            dataset = Dataset.from_iterable(
                [
                    DatasetItem(1, subset="a"),
                    DatasetItem(2, subset="b"),
                    DatasetItem(3, subset="c", media=Image(data=np.ones((2, 2, 3)))),
                    DatasetItem(4, subset="d", media=Image(data=np.ones((2, 3, 3)))),
                ],
                media_type=Image,
            )

            dataset.save(path, save_media=True)

            dataset.filter("/item[id >= 2]")
            dataset.transform("random_split", splits=(("train", 0.5), ("test", 0.5)), seed=42)
            dataset.save(save_media=True)

            self.assertEqual({"images", "annotations"}, set(os.listdir(path)))
            self.assertEqual(
                {"train.json", "test.json"}, set(os.listdir(osp.join(path, "annotations")))
            )
            self.assertEqual({"3.jpg"}, set(os.listdir(osp.join(path, "images", "train"))))
            self.assertEqual({"4.jpg"}, set(os.listdir(osp.join(path, "images", "test"))))
            self.assertEqual({"train", "c", "d", "test"}, set(os.listdir(osp.join(path, "images"))))
            self.assertEqual(set(), set(os.listdir(osp.join(path, "images", "c"))))
            self.assertEqual(set(), set(os.listdir(osp.join(path, "images", "d"))))
            compare_datasets(self, expected, Dataset.load(path))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_pointcloud(self):
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    subset="test",
                    media=PointCloud(
                        "1.pcd",
                        extra_images=[
                            Image(data=np.ones((5, 5, 3)), path="1/a.jpg"),
                            Image(data=np.ones((5, 4, 3)), path="1/b.jpg"),
                            Image(size=(5, 3), path="1/c.jpg"),
                        ],
                    ),
                    annotations=[
                        Cuboid3d(
                            [2, 2, 2],
                            [1, 1, 1],
                            [3, 3, 1],
                            id=1,
                            group=1,
                            label=0,
                            attributes={"x": True},
                        )
                    ],
                ),
            ],
            categories=["label"],
            media_type=PointCloud,
        )

        with TestDir() as test_dir:
            target_dataset = Dataset.from_iterable(
                [
                    DatasetItem(
                        id=1,
                        subset="test",
                        media=PointCloud(
                            osp.join(test_dir, "point_clouds", "test", "1.pcd"),
                            extra_images=[
                                Image(
                                    data=np.ones((5, 5, 3)),
                                    path=osp.join(
                                        test_dir, "related_images", "test", "1", "image_0.jpg"
                                    ),
                                ),
                                Image(
                                    data=np.ones((5, 4, 3)),
                                    path=osp.join(
                                        test_dir, "related_images", "test", "1", "image_1.jpg"
                                    ),
                                ),
                                Image(
                                    size=(5, 3),
                                    path=osp.join(
                                        test_dir, "related_images", "test", "1", "image_2.jpg"
                                    ),
                                ),
                            ],
                        ),
                        annotations=[
                            Cuboid3d(
                                [2, 2, 2],
                                [1, 1, 1],
                                [3, 3, 1],
                                id=1,
                                group=1,
                                label=0,
                                attributes={"x": True},
                            )
                        ],
                    ),
                ],
                categories=["label"],
                media_type=PointCloud,
            )
            self._test_save_and_load(
                source_dataset,
                partial(DatumaroConverter.convert, save_media=True),
                test_dir,
                target_dataset,
                compare=None,
                dimension=Dimensions.dim_3d,
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_save_and_load_with_skeleton(self):
        categories = {
            AnnotationType.label: LabelCategories.from_iterable(
                [
                    ("point2", "skeleton"),
                    ("point1", "skeleton"),
                    ("skeleton",),
                    ("point3", "skeleton"),
                    ("point4", "skeleton"),
                ]
            ),
            AnnotationType.points: PointsCategories.from_iterable(
                [
                    (2, ["point2", "point1", "point3", "point4"], {(2, 1), (0, 3)}),
                ]
            ),
        }
        source_elements = [
            Points([1, 1], label=0, attributes={"occluded": False, "outside": True}),
            Points(
                [2, 2],
                label=1,
                attributes={"occluded": False, "outside": False, "custom": "value"},
            ),
            Points([4, 4], label=4, attributes={"occluded": False, "outside": False}),
        ]
        source_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="img1",
                    subset="train",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[
                        Skeleton(
                            source_elements,
                            label=2,
                            attributes={"occluded": False},
                        )
                    ],
                ),
            ],
            categories=categories,
        )
        target_elements = [
            source_elements[0],
            source_elements[1],
            Points([0, 0], label=3, visibility=[Points.Visibility.absent.value]),
            source_elements[2],
        ]
        target_dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id="img1",
                    subset="train",
                    media=Image(data=np.ones((10, 10, 3))),
                    annotations=[
                        Skeleton(
                            target_elements,
                            label=2,
                            attributes={"occluded": False},
                        )
                    ],
                ),
            ],
            categories=categories,
        )

        with TestDir() as test_dir:
            self._test_save_and_load(
                source_dataset,
                partial(DatumaroConverter.convert, save_media=True),
                test_dir,
                target_dataset=target_dataset,
            )