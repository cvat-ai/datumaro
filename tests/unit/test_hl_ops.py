# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT
import json
import os

import numpy as np
import pytest

from datumaro import Dataset, DatasetItem, HLOps, Image
from datumaro.components.annotation import Bbox, Ellipse, Label, Polygon

from tests.requirements import Requirements, mark_requirement
from tests.utils.test_utils import TestCaseHelper, TestDir
from tests.utils.test_utils import compare_datasets as _compare_datasets


def compare_datasets(_, expected, actual):
    _compare_datasets(TestCaseHelper(), expected, actual)


class HLOpsTest:
    def test_can_transform(self):
        expected = Dataset.from_iterable(
            [DatasetItem(0, subset="train")], categories=["cat", "dog"]
        )

        dataset = Dataset.from_iterable(
            [DatasetItem(10, subset="train")], categories=["cat", "dog"]
        )

        actual = HLOps.transform(dataset, "reindex", start=0)

        compare_datasets(self, expected, actual)

    @pytest.mark.parametrize(
        "expr_or_filter_func",
        ["/item[id=0]", lambda item: str(item.id) == "0"],
        ids=["xpath", "pyfunc"],
    )
    def test_can_filter_items(self, expr_or_filter_func):
        expected = Dataset.from_iterable(
            [DatasetItem(0, subset="train")], categories=["cat", "dog"]
        )

        dataset = Dataset.from_iterable(
            [DatasetItem(0, subset="train"), DatasetItem(1, subset="train")],
            categories=["cat", "dog"],
        )

        actual = HLOps.filter(dataset, expr_or_filter_func)

        compare_datasets(self, expected, actual)

    @pytest.mark.parametrize(
        "expr_or_filter_func",
        ["/item/annotation[id=1]", lambda item, ann: str(ann.id) == "1"],
        ids=["xpath", "pyfunc"],
    )
    def test_can_filter_annotations(self, expr_or_filter_func):
        expected = Dataset.from_iterable(
            [DatasetItem(0, subset="train", annotations=[Label(0, id=1)])],
            categories=["cat", "dog"],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    subset="train",
                    annotations=[
                        Label(0, id=0),
                        Label(0, id=1),
                    ],
                ),
                DatasetItem(1, subset="train"),
            ],
            categories=["cat", "dog"],
        )

        actual = HLOps.filter(
            dataset, expr_or_filter_func, filter_annotations=True, remove_empty=True
        )

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_filter_by_annotation_types(self):
        annotations = [
            Label(0, id=0),
            Bbox(0, 0, 1, 1, id=1),
            Polygon([0, 0, 0, 1, 1, 1], id=2),
            Ellipse(0, 0, 1, 1, id=3),
        ]

        dataset = Dataset.from_iterable(
            [
                DatasetItem(
                    0,
                    subset="train",
                    annotations=annotations,
                )
            ],
        )

        types = {ann.type.name for ann in annotations}

        for t in types:
            allowed_types = types - {t}
            cmd = " or ".join([f"type='{allowed_type}'" for allowed_type in allowed_types])
            actual = HLOps.filter(
                dataset,
                f"/item/annotation[{cmd}]",
                filter_annotations=True,
                remove_empty=True,
            )
            actual_anns = [item for item in actual][0].annotations
            assert len(actual_anns) == len(allowed_types)

    def test_can_merge(self):
        expected = Dataset.from_iterable(
            [DatasetItem(0, subset="train"), DatasetItem(1, subset="train")],
            categories=["cat", "dog"],
        )

        dataset_a = Dataset.from_iterable(
            [
                DatasetItem(0, subset="train"),
            ],
            categories=["cat", "dog"],
        )

        dataset_b = Dataset.from_iterable(
            [DatasetItem(1, subset="train")], categories=["cat", "dog"]
        )

        actual = HLOps.merge(dataset_a, dataset_b)

        compare_datasets(self, expected, actual)

    def test_can_export(self):
        expected = Dataset.from_iterable(
            [DatasetItem(0, subset="train"), DatasetItem(1, subset="train")],
            categories=["cat", "dog"],
        )

        dataset = Dataset.from_iterable(
            [DatasetItem(0, subset="train"), DatasetItem(1, subset="train")],
            categories=["cat", "dog"],
        )

        with TestDir() as test_dir:
            HLOps.export(dataset, test_dir, "datumaro")
            actual = Dataset.load(test_dir)

            compare_datasets(self, expected, actual)

    def test_aggregate(self):
        expected = Dataset.from_iterable(
            [
                DatasetItem(0, subset="default"),
                DatasetItem(1, subset="default"),
                DatasetItem(2, subset="default"),
            ],
            categories=["cat", "dog"],
        )

        dataset = Dataset.from_iterable(
            [
                DatasetItem(0, subset="train"),
                DatasetItem(1, subset="val"),
                DatasetItem(2, subset="test"),
            ],
            categories=["cat", "dog"],
        )

        actual = HLOps.aggregate(
            dataset, from_subsets=["train", "val", "test"], to_subset="default"
        )

        compare_datasets(self, expected, actual)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_compare_equality(self):
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    annotations=[
                        Bbox(1, 2, 3, 4, label=0),
                    ],
                ),
                DatasetItem(id=200, subset="train"),
            ],
            categories=["a", "b"],
        )

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1),
                        Bbox(5, 6, 7, 8, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )
        with TestDir() as test_dir:
            HLOps.compare(dataset1, dataset2, method="equality", report_dir=test_dir)
            report_file = os.path.join(test_dir, "equality_compare.json")
            assert os.path.exists(report_file)
            with open(report_file, "r") as f:
                report = json.load(f)
            assert report["a_extra_items"] == [["200", "train"]]
            assert report["b_extra_items"] == []
            assert report["mismatches"] == []
            assert len(report["errors"]) == 1
            assert report["errors"][0]["type"] == "labels"

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_compare_table(self):
        dataset1 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=0),
                    ],
                ),
                DatasetItem(id=200, media=Image.from_numpy(np.ones((10, 15, 3))), subset="train"),
            ],
            categories=["a", "b"],
        )

        dataset2 = Dataset.from_iterable(
            [
                DatasetItem(
                    id=100,
                    subset="train",
                    media=Image.from_numpy(np.ones((10, 15, 3))),
                    annotations=[
                        Bbox(1, 2, 3, 4, label=1),
                        Bbox(5, 6, 7, 8, label=2),
                    ],
                ),
            ],
            categories=["a", "b", "c"],
        )
        with TestDir() as test_dir:
            HLOps.compare(dataset1, dataset2, method="table", report_dir=test_dir)
            assert os.path.exists(os.path.join(test_dir, "table_compare.json"))
            assert os.path.exists(os.path.join(test_dir, "table_compare.txt"))
            with open(os.path.join(test_dir, "table_compare.json"), "r") as f:
                report = json.load(f)
            assert report["high_level"] == {
                "Number of classes": ["2", "3"],
                "Common classes": ["a, b", "a, b"],
                "Classes": ["a, b", "a, b, c"],
                "Images count": ["2", "1"],
                "Unique images count": ["1", "1"],
                "Repeated images count": ["1", "0"],
                "Annotations count": ["1", "2"],
                "Unannotated images count": ["1", "0"],
            }
