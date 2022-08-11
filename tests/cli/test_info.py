# Copyright (C) 2022 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import io
import os.path as osp
from pathlib import Path

import pytest

from datumaro.util.test_utils import run_datum

from ..requirements import Requirements, mark_requirement

DUMMY_DATASET_DIR = osp.join(
    __file__[: __file__.rfind(osp.join("tests", ""))],
    "tests",
    "assets",
    "coco_dataset",
    "coco_instances",
)


class InfoTest:
    @pytest.fixture(autouse=True)
    def setup(
        self,
        fxt_stdout: io.StringIO,
        tmp_path: Path,
    ):
        self.tmp_path = tmp_path
        self.stdout = fxt_stdout

        yield

        self.tmp_path = None

    def run(self, cmd: str, *args: str, expected_code: int = 0) -> str:
        run_datum(
            self,
            cmd,
            *args,
            expected_code=expected_code,
        )
        return self.stdout.getvalue()

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_print_info_for_dataset(self):
        stdout = self.run("info", DUMMY_DATASET_DIR)

        assert "format: coco" in stdout
        assert "media type: image" in stdout
        assert "subsets" in stdout
