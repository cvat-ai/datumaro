# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from datumaro import Importer
from datumaro.components.format_detection import FormatDetectionContext


class YoloImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("obj.data")

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".data", "yolo")
