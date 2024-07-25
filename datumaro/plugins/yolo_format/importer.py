# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, Dict, List

import yaml

from datumaro import Importer
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.plugins.yolo_format.extractor import (
    Yolo8Extractor,
    Yolo8ObbExtractor,
    Yolo8PoseExtractor,
    Yolo8SegmentationExtractor,
)
from datumaro.plugins.yolo_format.format import Yolo8PoseFormat


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
        with context.probe_text_file(
            "data.yaml",
            f"must not have '{Yolo8PoseFormat.KPT_SHAPE_FIELD_NAME}' field",
        ) as f:
            try:
                config = yaml.safe_load(f)
                if Yolo8PoseFormat.KPT_SHAPE_FIELD_NAME in config:
                    raise Exception
            except yaml.YAMLError:
                raise Exception

    @classmethod
    def find_sources(cls, path) -> List[Dict[str, Any]]:
        return cls._find_sources_recursive(path, ".yaml", cls.EXTRACTOR.NAME)


class Yolo8SegmentationImporter(Yolo8Importer):
    EXTRACTOR = Yolo8SegmentationExtractor


class Yolo8ObbImporter(Yolo8Importer):
    EXTRACTOR = Yolo8ObbExtractor


class Yolo8PoseImporter(Yolo8Importer):
    EXTRACTOR = Yolo8PoseExtractor

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("data.yaml")
        with context.probe_text_file(
            "data.yaml",
            f"must have '{Yolo8PoseFormat.KPT_SHAPE_FIELD_NAME}' field",
        ) as f:
            try:
                config = yaml.safe_load(f)
                if Yolo8PoseFormat.KPT_SHAPE_FIELD_NAME not in config:
                    raise Exception
            except yaml.YAMLError:
                raise Exception
