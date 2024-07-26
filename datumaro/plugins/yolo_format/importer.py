# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from os import path as osp
from typing import Any, Dict, List

import yaml

from datumaro import Importer
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.plugins.yolo_format.extractor import (
    Yolo8Extractor,
    Yolo8OrientedBoxesExtractor,
    Yolo8PoseExtractor,
    Yolo8SegmentationExtractor,
)
from datumaro.plugins.yolo_format.format import Yolo8Path, Yolo8PoseFormat


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
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--config-file",
            help="The name of the file to read dataset config from",
        )
        return parser

    @classmethod
    def _check_config_file(cls, context, config_file):
        with context.probe_text_file(
            config_file,
            f"must not have '{Yolo8PoseFormat.KPT_SHAPE_FIELD_NAME}' field",
        ) as f:
            try:
                config = yaml.safe_load(f)
                if Yolo8PoseFormat.KPT_SHAPE_FIELD_NAME in config:
                    raise Exception
            except yaml.YAMLError:
                raise Exception

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f"*{Yolo8Path.CONFIG_FILE_EXT}")
        sources = cls.find_sources_with_params(context.root_path)
        if not sources or len(sources) > 1:
            context.fail("Cannot choose config file")

        cls._check_config_file(context, osp.relpath(sources[0]["url"], context.root_path))

    @classmethod
    def find_sources_with_params(
        cls, path, config_file=None, **extra_params
    ) -> List[Dict[str, Any]]:
        sources = cls._find_sources_recursive(
            path, Yolo8Path.CONFIG_FILE_EXT, cls.EXTRACTOR.NAME, max_depth=1
        )

        if config_file:
            return [source for source in sources if source["url"] == osp.join(path, config_file)]
        if len(sources) <= 1:
            return sources
        return [
            source
            for source in sources
            if source["url"] == osp.join(path, Yolo8Path.DEFAULT_CONFIG_FILE)
        ]


class Yolo8SegmentationImporter(Yolo8Importer):
    EXTRACTOR = Yolo8SegmentationExtractor


class Yolo8OrientedBoxesImporter(Yolo8Importer):
    EXTRACTOR = Yolo8OrientedBoxesExtractor


class Yolo8PoseImporter(Yolo8Importer):
    EXTRACTOR = Yolo8PoseExtractor

    @classmethod
    def _check_config_file(cls, context, config_file):
        with context.probe_text_file(
            config_file,
            f"must have '{Yolo8PoseFormat.KPT_SHAPE_FIELD_NAME}' field",
        ) as f:
            try:
                config = yaml.safe_load(f)
                if Yolo8PoseFormat.KPT_SHAPE_FIELD_NAME not in config:
                    raise Exception
            except yaml.YAMLError:
                raise Exception
