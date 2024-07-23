# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, Dict, List

from datumaro import Importer
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.plugins.yolo_format.extractor import (
    Yolo8Extractor,
    Yolo8ObbExtractor,
    Yolo8SegmentationExtractor,
)


class YoloImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("obj.data")

    @classmethod
    def find_sources(cls, path) -> List[Dict[str, Any]]:
        return cls._find_sources_recursive(path, ".data", "yolo")


class Yolo8Importer(Importer):
    EXTRACTOR = Yolo8Extractor

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("data.yaml")

    @classmethod
    def find_sources(cls, path) -> List[Dict[str, Any]]:
        return cls._find_sources_recursive(path, ".yaml", cls.EXTRACTOR.NAME)


class Yolo8SegmentationImporter(Yolo8Importer):
    EXTRACTOR = Yolo8SegmentationExtractor


class Yolo8ObbImporter(Yolo8Importer):
    EXTRACTOR = Yolo8ObbExtractor
