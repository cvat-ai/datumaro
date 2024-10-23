import os.path as osp
from unittest import TestCase

import numpy as np

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
)
from datumaro.components.dataset import Dataset
from datumaro.components.extractor import DEFAULT_SUBSET_NAME, DatasetItem
from datumaro.components.media import Image, MultiframeImage, PointCloud
from datumaro.components.operations import (
    FailedAttrVotingError,
    IntersectMerge,
    NoMatchingAnnError,
    NoMatchingItemError,
    WrongGroupError,
    compute_ann_statistics,
    compute_image_statistics,
    find_unique_images,
    match_segments,
    mean_std,
)

from tests.requirements import Requirements, mark_requirement
from tests.utils.assets import get_test_asset_path
from tests.utils.test_utils import compare_datasets


class TestOperations(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_mean_std(self):
        expected_mean = [100, 50, 150]
        expected_std = [20, 50, 10]

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=i,
                    media=Image(data=np.random.normal(expected_mean, expected_std, size=(h, w, 3))),
                )
                for i, (w, h) in enumerate([(3000, 100), (800, 600), (400, 200), (700, 300)])
            ]
        )

        actual_mean, actual_std = mean_std(dataset)

        for em, am in zip(expected_mean, actual_mean):
            self.assertAlmostEqual(em, am, places=0)
        for estd, astd in zip(expected_std, actual_std):
            self.assertAlmostEqual(estd, astd, places=0)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_image_stats(self):
        expected_mean = [100, 50, 150]
        expected_std = [20, 50, 10]

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=i,
                    media=Image(data=np.random.normal(expected_mean, expected_std, size=(h, w, 3))),
                )
                for i, (w, h) in enumerate([(3000, 100), (800, 600), (400, 200), (700, 300)])
            ]
        )
        dataset.put(dataset.get("1"), id="5", subset="train")

        actual = compute_image_statistics(dataset)

        self.assertEqual(
            actual["dataset"],
            {
                "images count": 5,
                "unique images count": 4,
                "repeated images count": 1,
                "repeated images": [[("1", "default"), ("5", "train")]],
            },
        )
        self.assertEqual(actual["subsets"]["default"]["images count"], 4)
        self.assertEqual(actual["subsets"]["train"]["images count"], 1)

        actual_mean = actual["subsets"]["default"]["image mean"][::-1]
        actual_std = actual["subsets"]["default"]["image std"][::-1]
        for em, am in zip(expected_mean, actual_mean):
            self.assertAlmostEqual(em, am, places=0)
        for estd, astd in zip(expected_std, actual_std):
            self.assertAlmostEqual(estd, astd, places=0)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_image_stats_with_no_image_infos(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=0, media=Image(size=(10, 10))),
                DatasetItem(id=1, media=Image(path="inexistent.path")),
                DatasetItem(id=2),
            ]
        )

        actual = compute_image_statistics(dataset)

        self.assertEqual(
            actual["dataset"],
            {
                "images count": 3,
                "unique images count": 3,
                "repeated images count": 0,
                "repeated images": [],
            },
        )
        self.assertEqual("n/a", actual["subsets"]["default"]["image mean"])
        self.assertEqual("n/a", actual["subsets"]["default"]["image std"])

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_stats(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    id=1,
                    media=Image(data=np.ones((5, 5, 3))),
                    annotations=[
                        Caption("hello"),
                        Caption("world"),
                        Label(
                            2,
                            attributes={
                                "x": 1,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            2,
                            2,
                            label=2,
                            attributes={
                                "score": 0.5,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            2,
                            2,
                            attributes={
                                "x": 1,
                                "y": "3",
                                "occluded": True,
                            },
                        ),
                        Points([1, 2, 2, 0, 1, 1], label=0),
                        Mask(
                            label=3,
                            image=np.array(
                                [
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 1, 1, 1],
                                    [0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0],
                                ]
                            ),
                        ),
                    ],
                ),
                DatasetItem(
                    id=2,
                    media=Image(data=np.ones((2, 4, 3))),
                    annotations=[
                        Label(
                            2,
                            attributes={
                                "x": 2,
                                "y": "2",
                            },
                        ),
                        Bbox(
                            1,
                            2,
                            2,
                            2,
                            label=3,
                            attributes={
                                "score": 0.5,
                            },
                        ),
                        Bbox(
                            5,
                            6,
                            2,
                            2,
                            attributes={
                                "x": 2,
                                "y": "3",
                                "occluded": False,
                            },
                        ),
                    ],
                ),
                DatasetItem(id=3),
                DatasetItem(id="2.2", media=Image(data=np.ones((2, 4, 3)))),
            ],
            categories=["label_%s" % i for i in range(4)],
        )

        expected = {
            "images count": 4,
            "annotations count": 10,
            "unannotated images count": 2,
            "unannotated images": ["3", "2.2"],
            "annotations by type": {
                "label": {
                    "count": 2,
                },
                "polygon": {
                    "count": 0,
                },
                "polyline": {
                    "count": 0,
                },
                "bbox": {
                    "count": 4,
                },
                "mask": {
                    "count": 1,
                },
                "points": {
                    "count": 1,
                },
                "caption": {
                    "count": 2,
                },
                "cuboid_3d": {"count": 0},
                "super_resolution_annotation": {"count": 0},
                "depth_annotation": {"count": 0},
                "skeleton": {"count": 0},
            },
            "annotations": {
                "labels": {
                    "count": 6,
                    "distribution": {
                        "label_0": [1, 1 / 6],
                        "label_1": [0, 0.0],
                        "label_2": [3, 3 / 6],
                        "label_3": [2, 2 / 6],
                    },
                    "attributes": {
                        "x": {
                            "count": 2,  # annotations with no label are skipped
                            "values count": 2,
                            "values present": ["1", "2"],
                            "distribution": {
                                "1": [1, 1 / 2],
                                "2": [1, 1 / 2],
                            },
                        },
                        "y": {
                            "count": 2,  # annotations with no label are skipped
                            "values count": 1,
                            "values present": ["2"],
                            "distribution": {
                                "2": [2, 2 / 2],
                            },
                        },
                        # must not include "special" attributes like "occluded"
                    },
                },
                "segments": {
                    "avg. area": (4 * 2 + 9 * 1) / 3,
                    "area distribution": [
                        {"min": 4.0, "max": 4.5, "count": 2, "percent": 2 / 3},
                        {"min": 4.5, "max": 5.0, "count": 0, "percent": 0.0},
                        {"min": 5.0, "max": 5.5, "count": 0, "percent": 0.0},
                        {"min": 5.5, "max": 6.0, "count": 0, "percent": 0.0},
                        {"min": 6.0, "max": 6.5, "count": 0, "percent": 0.0},
                        {"min": 6.5, "max": 7.0, "count": 0, "percent": 0.0},
                        {"min": 7.0, "max": 7.5, "count": 0, "percent": 0.0},
                        {"min": 7.5, "max": 8.0, "count": 0, "percent": 0.0},
                        {"min": 8.0, "max": 8.5, "count": 0, "percent": 0.0},
                        {"min": 8.5, "max": 9.0, "count": 1, "percent": 1 / 3},
                    ],
                    "pixel distribution": {
                        "label_0": [0, 0.0],
                        "label_1": [0, 0.0],
                        "label_2": [4, 4 / 17],
                        "label_3": [13, 13 / 17],
                    },
                },
            },
        }

        actual = compute_ann_statistics(dataset)

        self.assertEqual(expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_stats_with_empty_dataset(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1),
                DatasetItem(id=3),
            ],
            categories=["label_%s" % i for i in range(4)],
        )

        expected = {
            "images count": 2,
            "annotations count": 0,
            "unannotated images count": 2,
            "unannotated images": ["1", "3"],
            "annotations by type": {
                "label": {
                    "count": 0,
                },
                "polygon": {
                    "count": 0,
                },
                "polyline": {
                    "count": 0,
                },
                "bbox": {
                    "count": 0,
                },
                "mask": {
                    "count": 0,
                },
                "points": {
                    "count": 0,
                },
                "caption": {
                    "count": 0,
                },
                "cuboid_3d": {"count": 0},
                "super_resolution_annotation": {"count": 0},
                "depth_annotation": {"count": 0},
                "skeleton": {"count": 0},
            },
            "annotations": {
                "labels": {
                    "count": 0,
                    "distribution": {
                        "label_0": [0, 0.0],
                        "label_1": [0, 0.0],
                        "label_2": [0, 0.0],
                        "label_3": [0, 0.0],
                    },
                    "attributes": {},
                },
                "segments": {
                    "avg. area": 0.0,
                    "area distribution": [],
                    "pixel distribution": {
                        "label_0": [0, 0.0],
                        "label_1": [0, 0.0],
                        "label_2": [0, 0.0],
                        "label_3": [0, 0.0],
                    },
                },
            },
        }

        actual = compute_ann_statistics(dataset)

        self.assertEqual(expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_unique_image_count(self):
        expected = {
            frozenset([("1", "a"), ("1", "b")]),
            frozenset([("2", DEFAULT_SUBSET_NAME), ("3", DEFAULT_SUBSET_NAME)]),
            frozenset([("4", DEFAULT_SUBSET_NAME)]),
        }

        dataset = Dataset.from_iterable(
            [
                # no image data, but the same path
                DatasetItem(1, subset="a", media=Image(path="1.jpg")),
                DatasetItem(1, subset="b", media=Image(path="1.jpg")),
                # same images
                DatasetItem(2, media=Image(data=np.array([1]))),
                DatasetItem(3, media=Image(data=np.array([1]))),
                # no image is always a unique image
                DatasetItem(4),
            ]
        )

        groups = find_unique_images(dataset)

        self.assertEqual(expected, set(frozenset(s) for s in groups.values()))


class TestAnnotationMatching(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_match_shape_first_and_label_later(self):
        # Labels must mismatch even though there is a better possible match around
        # when the default matchers are used. This yields more adequate matching
        # results for use cases such as annotation tools.
        # In the case of models, typically we work with annotations after NMS,
        # so this method will yield adequate results.
        # If there is no NMS, a different strategy can be considered,
        # which looks around for best matches including the labels.

        anns1 = [
            Bbox(0, 0, 4, 4, label=0, id=1),
            Bbox(1, 1, 4, 4, label=1, id=2),
        ]

        anns2 = [
            Bbox(1, 1, 4, 4, label=0, id=2),
            Bbox(0, 0, 4, 4, label=1, id=1),
        ]

        matches, mismatches, a_extra, b_extra = match_segments(anns1, anns2, dist_thresh=0.5)
        assert sorted(mismatches, key=lambda e: e[0].id) == [
            (anns1[0], anns2[1]),
            (anns1[1], anns2[0]),
        ]
        assert not matches + a_extra + b_extra

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_match(self):
        anns1 = [
            # mismatching
            Bbox(0, 0, 4, 4, label=0, id=1),
            Bbox(1, 1, 4, 4, label=1, id=2),
            # matching
            Bbox(5, 5, 4, 4, label=0, id=3),
            Bbox(6, 6, 4, 4, label=1, id=4),
            # extra
            Bbox(6, 0, 4, 4, label=1, id=5),
        ]

        anns2 = [
            # mismatching
            Bbox(1, 1, 4, 4, label=0, id=2),
            Bbox(0, 0, 4, 4, label=1, id=1),
            # matching
            Bbox(5, 5, 4, 4, label=0, id=3),
            Bbox(6, 6, 4, 4, label=1, id=4),
            # extra
            Bbox(0, 6, 4, 4, label=1, id=5),
        ]

        matches, mismatches, a_extra, b_extra = match_segments(anns1, anns2, dist_thresh=0.5)
        assert sorted(mismatches, key=lambda e: e[0].id) == [
            (anns1[0], anns2[1]),
            (anns1[1], anns2[0]),
        ]
        assert sorted(matches, key=lambda e: e[0].id) == [
            (anns1[2], anns2[2]),
            (anns1[3], anns2[3]),
        ]
        assert a_extra == [anns1[4]]
        assert b_extra == [anns2[4]]


class TestMultimerge(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_match_items(self):
        # items 1 and 3 are unique, item 2 is common and should be merged

        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                    ],
                ),
                DatasetItem(
                    2,
                    annotations=[
                        Label(0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    2,
                    annotations=[
                        Label(1),
                    ],
                ),
                DatasetItem(
                    3,
                    annotations=[
                        Label(0),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        source2 = Dataset.from_iterable(
            [
                DatasetItem(2, annotations=[Label(0), Bbox(1, 2, 3, 4)]),
            ],
            categories=["a", "b"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0, attributes={"score": 1 / 3}),
                    ],
                ),
                DatasetItem(
                    2,
                    annotations=[
                        Label(0, attributes={"score": 2 / 3}),
                        Label(1, attributes={"score": 1 / 3}),
                        Bbox(1, 2, 3, 4, attributes={"score": 1.0}),
                    ],
                ),
                DatasetItem(
                    3,
                    annotations=[
                        Label(0, attributes={"score": 1 / 3}),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        merger = IntersectMerge()
        merged = merger([source0, source1, source2])

        compare_datasets(self, expected, merged)
        self.assertEqual(
            [
                NoMatchingItemError(item_id=("1", DEFAULT_SUBSET_NAME), sources={1, 2}),
                NoMatchingItemError(item_id=("3", DEFAULT_SUBSET_NAME), sources={0, 2}),
            ],
            sorted(
                (e for e in merger.errors if isinstance(e, NoMatchingItemError)),
                key=lambda e: e.item_id,
            ),
        )
        self.assertEqual(
            [
                NoMatchingAnnError(
                    item_id=("2", DEFAULT_SUBSET_NAME),
                    sources={0, 1},
                    ann=source2.get("2").annotations[1],
                ),
            ],
            sorted(
                (e for e in merger.errors if isinstance(e, NoMatchingAnnError)),
                key=lambda e: e.item_id,
            ),
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_match_shapes(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        # unique
                        Bbox(1, 2, 3, 4, label=1),
                        # common
                        Mask(
                            label=2,
                            z_order=2,
                            image=np.array(
                                [
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [1, 1, 1, 0],
                                    [1, 1, 1, 0],
                                ]
                            ),
                        ),
                        Polygon([1, 0, 3, 2, 1, 2]),
                        # an instance with keypoints
                        Bbox(4, 5, 2, 4, label=2, z_order=1, group=1),
                        Points([5, 6], label=0, group=1),
                        Points([6, 8], label=1, group=1),
                        PolyLine([1, 1, 2, 1, 3, 1]),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        # common
                        Mask(
                            label=2,
                            image=np.array(
                                [
                                    [0, 0, 0, 0],
                                    [0, 1, 1, 1],
                                    [0, 1, 1, 1],
                                    [0, 1, 1, 1],
                                ]
                            ),
                        ),
                        Polygon([0, 2, 2, 0, 2, 1]),
                        # an instance with keypoints
                        Bbox(4, 4, 2, 5, label=2, z_order=1, group=2),
                        Points([5.5, 6.5], label=0, group=2),
                        Points([6, 8], label=1, group=2),
                        PolyLine([1, 1.5, 2, 1.5]),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        source2 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        # common
                        Mask(
                            label=2,
                            z_order=3,
                            image=np.array(
                                [
                                    [0, 0, 1, 1],
                                    [0, 1, 1, 1],
                                    [1, 1, 1, 1],
                                    [1, 1, 1, 0],
                                ]
                            ),
                        ),
                        Polygon([3, 1, 2, 2, 0, 1]),
                        # an instance with keypoints, one is missing
                        Bbox(3, 6, 2, 3, label=2, z_order=4, group=3),
                        Points([4.5, 5.5], label=0, group=3),
                        PolyLine([1, 1.25, 3, 1, 4, 2]),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        # unique
                        Bbox(1, 2, 3, 4, label=1),
                        # common
                        # nearest to mean bbox
                        Mask(
                            label=2,
                            z_order=3,
                            image=np.array(
                                [
                                    [0, 0, 0, 0],
                                    [0, 1, 1, 1],
                                    [0, 1, 1, 1],
                                    [0, 1, 1, 1],
                                ]
                            ),
                        ),
                        Polygon([1, 0, 3, 2, 1, 2]),
                        # an instance with keypoints
                        Bbox(4, 5, 2, 4, label=2, z_order=4, group=1),
                        Points([5, 6], label=0, group=1),
                        Points([6, 8], label=1, group=1),
                        PolyLine([1, 1.25, 3, 1, 4, 2]),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        merger = IntersectMerge(conf={"quorum": 1, "pairwise_dist": 0.1})
        merged = merger([source0, source1, source2])

        compare_datasets(self, expected, merged, ignored_attrs={"score"})
        self.assertEqual(
            [
                NoMatchingAnnError(
                    item_id=("1", DEFAULT_SUBSET_NAME),
                    sources={2},
                    ann=source0.get("1").annotations[5],
                ),
                NoMatchingAnnError(
                    item_id=("1", DEFAULT_SUBSET_NAME),
                    sources={1, 2},
                    ann=source0.get("1").annotations[0],
                ),
            ],
            sorted(
                (e for e in merger.errors if isinstance(e, NoMatchingAnnError)),
                key=lambda e: len(e.sources),
            ),
        )

    @mark_requirement(Requirements.DATUM_BUG_219)
    def test_can_match_lines_when_line_not_approximated(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        PolyLine([1, 1, 2, 1, 3, 5, 5, 5, 8, 3]),
                    ],
                ),
            ]
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        PolyLine([1, 1, 8, 3]),
                    ],
                ),
            ]
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        PolyLine([1, 1, 2, 1, 3, 5, 5, 5, 8, 3]),
                    ],
                ),
            ],
            categories=[],
        )

        merger = IntersectMerge(conf={"quorum": 1, "pairwise_dist": 0.1})
        merged = merger([source0, source1])

        compare_datasets(self, expected, merged, ignored_attrs={"score"})
        self.assertEqual(0, len(merger.errors))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_attributes(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(
                            2,
                            attributes={
                                "unique": 1,
                                "common_under_quorum": 2,
                                "common_over_quorum": 3,
                                "ignored": "q",
                            },
                        ),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(
                            2,
                            attributes={
                                "common_under_quorum": 2,
                                "common_over_quorum": 3,
                                "ignored": "q",
                            },
                        ),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        source2 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(
                            2,
                            attributes={
                                "common_over_quorum": 3,
                                "ignored": "q",
                            },
                        ),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(2, attributes={"common_over_quorum": 3}),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        merger = IntersectMerge(conf={"quorum": 3, "ignored_attributes": {"ignored"}})
        merged = merger([source0, source1, source2])

        compare_datasets(self, expected, merged, ignored_attrs={"score"})
        self.assertEqual(2, len([e for e in merger.errors if isinstance(e, FailedAttrVotingError)]))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_group_checks(self):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Bbox(0, 0, 0, 0, label=0, group=1),  # misses an optional label
                        Bbox(0, 0, 0, 0, label=1, group=1),
                        Bbox(0, 0, 0, 0, label=2, group=2),  # misses a mandatory label - error
                        Bbox(0, 0, 0, 0, label=2, group=2),
                        Bbox(0, 0, 0, 0, label=4),  # misses an optional label
                        Bbox(0, 0, 0, 0, label=5),  # misses a mandatory label - error
                        Bbox(0, 0, 0, 0, label=0),  # misses a mandatory label - error
                        Bbox(0, 0, 0, 0, label=3),  # not listed - not checked
                    ],
                ),
            ],
            categories=["a", "a_g1", "a_g2_opt", "b", "c", "c_g1_opt"],
        )

        merger = IntersectMerge(conf={"groups": [["a", "a_g1", "a_g2_opt?"], ["c", "c_g1_opt?"]]})
        merger([dataset, dataset])

        self.assertEqual(
            3, len([e for e in merger.errors if isinstance(e, WrongGroupError)]), merger.errors
        )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_classes(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                        Label(1),
                        Bbox(0, 0, 1, 1, label=1),
                    ],
                ),
            ],
            categories=["a", "b"],
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                        Label(1),
                        Bbox(0, 0, 1, 1, label=0),
                        Bbox(0, 0, 1, 1, label=1),
                    ],
                ),
            ],
            categories=["b", "c"],
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                        Label(1),
                        Label(2),
                        Bbox(0, 0, 1, 1, label=1),
                        Bbox(0, 0, 1, 1, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )

        merger = IntersectMerge()
        merged = merger([source0, source1])

        compare_datasets(self, expected, merged, ignored_attrs={"score"})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_categories(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b"]),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (0, ["l0", "l1"]),
                        (1, ["l2", "l3"]),
                    ]
                ),
                AnnotationType.mask: MaskCategories(
                    {
                        0: (0, 1, 2),
                        1: (1, 2, 3),
                    }
                ),
            },
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["c", "b"]),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (0, []),
                        (1, ["l2", "l3"]),
                    ]
                ),
                AnnotationType.mask: MaskCategories(
                    {
                        0: (0, 2, 4),
                        1: (1, 2, 3),
                    }
                ),
            },
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(
                    1,
                    annotations=[
                        Label(0),
                        Label(2),
                    ],
                ),
            ],
            categories={
                AnnotationType.label: LabelCategories.from_iterable(["a", "b", "c"]),
                AnnotationType.points: PointsCategories.from_iterable(
                    [
                        (0, ["l0", "l1"]),
                        (1, ["l2", "l3"]),
                        (2, []),
                    ]
                ),
                AnnotationType.mask: MaskCategories(
                    {
                        0: (0, 1, 2),
                        1: (1, 2, 3),
                        2: (0, 2, 4),
                    }
                ),
            },
        )

        merger = IntersectMerge()
        merged = merger([source0, source1])

        compare_datasets(self, expected, merged, ignored_attrs={"score"})

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_point_clouds(self):
        dataset_dir = get_test_asset_path("sly_pointcloud_dataset")
        pcd1 = osp.join(dataset_dir, "ds0", "pointcloud", "frame1.pcd")
        pcd2 = osp.join(dataset_dir, "ds0", "pointcloud", "frame2.pcd")

        image1 = Image(
            path=osp.join(dataset_dir, "ds0", "related_images", "frame1_pcd", "img2.png")
        )
        image2 = Image(
            path=osp.join(dataset_dir, "ds0", "related_images", "frame2_pcd", "img1.png")
        )

        source0 = Dataset.from_iterable(
            [
                DatasetItem(1, media=PointCloud(path=pcd1, extra_images=[image1])),
                DatasetItem(2, media=PointCloud(path=pcd1, extra_images=[image1])),
                DatasetItem(3, media=PointCloud(path=pcd2)),
                DatasetItem(4),
                DatasetItem(5, media=PointCloud(path=pcd2)),
            ],
            categories=[],
            media_type=PointCloud,
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(1, media=PointCloud(path=pcd1, extra_images=[image1])),
                DatasetItem(2, media=PointCloud(path=pcd1, extra_images=[image2])),
                DatasetItem(3),
                DatasetItem(4, media=PointCloud(path=pcd2)),
                DatasetItem(5, media=PointCloud(path=pcd2, extra_images=[image2])),
            ],
            categories=[],
            media_type=PointCloud,
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(1, media=PointCloud(path=pcd1, extra_images=[image1])),
                DatasetItem(2, media=PointCloud(path=pcd1, extra_images=[image1, image2])),
                DatasetItem(3, media=PointCloud(path=pcd2)),
                DatasetItem(4, media=PointCloud(path=pcd2)),
                DatasetItem(5, media=PointCloud(path=pcd2, extra_images=[image2])),
            ],
            categories=[],
            media_type=PointCloud,
        )

        merger = IntersectMerge()
        merged = merger([source0, source1])

        compare_datasets(self, expected, merged)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_merge_multiframe_images(self):
        source0 = Dataset.from_iterable(
            [
                DatasetItem(1, media=MultiframeImage([np.ones((1, 5, 3))] * 2)),
                DatasetItem(2, media=MultiframeImage([np.ones((3, 5, 3))] * 2)),
                DatasetItem(3, media=MultiframeImage([np.zeros((1, 5, 3))] * 2)),
                DatasetItem(4),
                DatasetItem(5, media=MultiframeImage([np.ones((1, 5, 3))] * 4)),
            ],
            categories=[],
            media_type=MultiframeImage,
        )

        source1 = Dataset.from_iterable(
            [
                DatasetItem(1, media=MultiframeImage([np.ones((1, 5, 3))] * 2)),
                DatasetItem(2, media=MultiframeImage([np.ones((3, 5, 3))] * 3)),
                DatasetItem(3),
                DatasetItem(4, media=MultiframeImage([np.ones((4, 5, 3))] * 2)),
                DatasetItem(5, media=MultiframeImage([np.ones((1, 5, 3))] * 2)),
            ],
            categories=[],
            media_type=MultiframeImage,
        )

        expected = Dataset.from_iterable(
            [
                DatasetItem(1, media=MultiframeImage([np.ones((1, 5, 3))] * 2)),
                DatasetItem(2, media=MultiframeImage([np.ones((3, 5, 3))] * 3)),
                DatasetItem(3, media=MultiframeImage([np.zeros((1, 5, 3))] * 2)),
                DatasetItem(4, media=MultiframeImage([np.ones((4, 5, 3))] * 2)),
                DatasetItem(5, media=MultiframeImage([np.ones((1, 5, 3))] * 4)),
            ],
            categories=[],
            media_type=MultiframeImage,
        )

        merger = IntersectMerge()
        merged = merger([source0, source1])

        compare_datasets(self, expected, merged)