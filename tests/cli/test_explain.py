# Copyright (C) 2022 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import os
import os.path as osp
import textwrap
from typing import Sequence

import numpy as np

from datumaro.components.extractor import DatasetItem
from datumaro.components.media import Image
from datumaro.components.project import Dataset, Project
from datumaro.util.image import load_image
from datumaro.util.scope import scope_add, scoped
from datumaro.util.test_utils import TestDir
from datumaro.util.test_utils import run_datum as run

from ..requirements import Requirements, mark_requirement


class ExplainTest:
    def _check_heatmap(self, heatmap: np.ndarray, roi: Sequence[int]):
        # check that heatmaps have a significant amount of nonzero values in the ROI area
        hm_sum = np.sum(heatmap)
        hm_area = np.prod(heatmap.shape)
        roi_sum = np.sum(heatmap[roi[0] : roi[1], roi[2] : roi[3]])
        roi_area = (roi[1] - roi[0]) * (roi[3] - roi[2])
        roi_den = roi_sum / roi_area
        hrest_den = (hm_sum - roi_sum) / (hm_area - roi_area)
        assert 0 < roi_sum
        assert hrest_den < roi_den

    def _create_launcher_plugin(self, dst_dir):
        launcher_code = textwrap.dedent(
            """
            import numpy as np
            import datumaro as dm

            class MyLauncher(dm.Launcher):
                def __init__(self, class_count, roi, **kwargs):
                    self.class_count = class_count
                    self.roi = roi

                def launch(self, inputs):
                    for inp in inputs:
                        yield self._process(inp)

                def _process(self, image):
                    roi = self.roi
                    roi_area = (roi[1] - roi[0]) * (roi[3] - roi[2])
                    cls_ratio = np.sum(image[roi[0] : roi[1], roi[2] : roi[3], 0]) / roi_area
                    if 0.5 < cls_ratio:
                        cls = 0
                        cls_conf = cls_ratio
                    else:
                        cls = 1
                        cls_conf = 1.0 - cls_ratio

                    other_conf = (1.0 - cls_conf) / (self.class_count - 1)

                    return [
                        # must return score for all classes
                        dm.Label(i, attributes={"score": cls_conf if cls == i else other_conf})
                        for i in range(self.class_count)
                    ]
            """
        )
        os.makedirs(dst_dir, exist_ok=True)
        launcher_url = osp.join(dst_dir, "__init__.py")
        with open(launcher_url, "w") as f:
            f.write(launcher_code)

    def _create_dataset(self, dataset_url):
        dataset = Dataset.from_iterable(
            [
                DatasetItem(id=1, media=Image(np.ones((20, 20, 3)))),
                DatasetItem(id=2, media=Image(np.ones((15, 15, 3)))),
                DatasetItem(id=3, media=Image(np.ones((16, 18, 3)))),
            ],
            categories=["a", "b"],
        )
        dataset.save(dataset_url, save_media=True)

        return dataset

    def _create_project(self, test_dir, dataset_url, model_name, roi):
        launcher_url = osp.join(test_dir, "launcher_plugin")
        self._create_launcher_plugin(launcher_url)

        proj_dir = osp.join(test_dir, "proj")
        with Project.init(proj_dir) as project:
            project.add_plugin("my", source=launcher_url)
            project.add_model(model_name, launcher="my", options={"class_count": 2, "roi": roi})
            project.import_source("source", dataset_url, "datumaro")
            project.save()

        return project

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_run_explain_on_dataset_with_custom_model(self):
        test_dir = scope_add(TestDir())

        dataset_url = osp.join(test_dir, "dataset")
        dataset = self._create_dataset(dataset_url)

        proj_dir = osp.join(test_dir, "project")
        roi = [2, 10, 2, 6]
        model_name = "mymodel"
        project = self._create_project(proj_dir, dataset_url, model_name=model_name, roi=roi)

        result_dir = osp.join(test_dir, "result")
        run(
            self,
            "explain",
            "-p",
            proj_dir,
            "-o",
            result_dir,
            "-m",
            model_name,
            "rise",
            "--max-samples",
            "100",
            "--mw",
            "2",
            "--mh",
            "2",
        )

        expected_items = {
            # classification models produce only 1 heatmap per image
            img_id: f"{img_id}-heatmap-0.png"
            for img_id in [item.id for item in dataset]
        }
        assert set(os.listdir(result_dir)) == set(expected_items.values())

        for fname in expected_items.values():
            heatmap = load_image(osp.join(result_dir, fname))
            self._check_heatmap(heatmap, roi)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    @scoped
    def test_can_run_explain_on_image_with_custom_model(self):
        test_dir = scope_add(TestDir())

        dataset_url = osp.join(test_dir, "dataset")
        dataset = self._create_dataset(dataset_url)

        proj_dir = osp.join(test_dir, "project")
        roi = [2, 10, 2, 6]
        model_name = "mymodel"
        project = self._create_project(proj_dir, dataset_url, model_name=model_name, roi=roi)

        result_dir = osp.join(test_dir, "result")
        run(
            self,
            "explain",
            "-p",
            proj_dir,
            osp.join(dataset_url, "images", "1.jpg"),
            "-o",
            result_dir,
            "-m",
            model_name,
            "rise",
            "--max-samples",
            "100",
            "--mw",
            "2",
            "--mh",
            "2",
        )

        expected_items = {
            # classification models produce only 1 heatmap per image
            img_id: f"{img_id}-heatmap-0.png"
            for img_id in [
                "0",
            ]
        }
        assert set(os.listdir(result_dir)) == set(expected_items.values())

        for fname in expected_items.values():
            heatmap = load_image(osp.join(result_dir, fname))
            self._check_heatmap(heatmap, roi)
