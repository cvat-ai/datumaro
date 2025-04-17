# Copyright (C) 2023 Intel Corporation
# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
from os import path as osp
from typing import Any, Dict, List

import yaml

from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.components.importer import Importer
from datumaro.plugins.data_formats.yolo.base import (
    YoloUltralyticsClassificationBase,
    YoloUltralyticsDetectionBase,
    YoloUltralyticsOrientedBoxesBase,
    YoloUltralyticsPoseBase,
    YoloUltralyticsSegmentationBase,
)
from datumaro.plugins.data_formats.yolo.format import (
    YoloUltralyticsClassificationFormat,
    YoloUltralyticsPath,
    YoloUltralyticsPoseFormat,
)
from datumaro.util.image import contains_only_images
from datumaro.util.meta_file_util import DATASET_META_FILE


class YoloImporter(Importer):
    DETECT_CONFIDENCE = FormatDetectionConfidence.MEDIUM

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("obj.data")

    @classmethod
    def find_sources(cls, path) -> List[Dict[str, Any]]:
        return cls._find_sources_recursive(path, ".data", "yolo")

    @property
    def can_stream(self) -> bool:
        return True


class YoloUltralyticsDetectionImporter(Importer):
    EXTRACTOR = YoloUltralyticsDetectionBase

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
            f"must not have '{YoloUltralyticsPoseFormat.KPT_SHAPE_FIELD_NAME}' field",
        ) as f:
            try:
                config = yaml.safe_load(f)
                if YoloUltralyticsPoseFormat.KPT_SHAPE_FIELD_NAME in config:
                    raise Exception
            except yaml.YAMLError:
                raise Exception

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file(f"*{YoloUltralyticsPath.CONFIG_FILE_EXT}")
        sources = cls.find_sources_with_params(context.root_path)
        if not sources or len(sources) > 1:
            context.fail("Cannot choose config file")

        cls._check_config_file(context, osp.relpath(sources[0]["url"], context.root_path))

    @classmethod
    def find_sources_with_params(
        cls, path, config_file=None, **extra_params
    ) -> List[Dict[str, Any]]:
        sources = cls._find_sources_recursive(
            path, YoloUltralyticsPath.CONFIG_FILE_EXT, cls.EXTRACTOR.NAME, max_depth=1
        )

        if config_file:
            return [source for source in sources if source["url"] == osp.join(path, config_file)]
        if len(sources) <= 1:
            return sources
        return [
            source
            for source in sources
            if source["url"] == osp.join(path, YoloUltralyticsPath.DEFAULT_CONFIG_FILE)
        ]

    @property
    def can_stream(self) -> bool:
        return True


class YoloUltralyticsSegmentationImporter(YoloUltralyticsDetectionImporter):
    EXTRACTOR = YoloUltralyticsSegmentationBase


class YoloUltralyticsOrientedBoxesImporter(YoloUltralyticsDetectionImporter):
    EXTRACTOR = YoloUltralyticsOrientedBoxesBase


class YoloUltralyticsPoseImporter(YoloUltralyticsDetectionImporter):
    EXTRACTOR = YoloUltralyticsPoseBase

    @classmethod
    def _check_config_file(cls, context, config_file):
        with context.probe_text_file(
            config_file,
            f"must have '{YoloUltralyticsPoseFormat.KPT_SHAPE_FIELD_NAME}' field",
        ) as f:
            try:
                config = yaml.safe_load(f)
                if YoloUltralyticsPoseFormat.KPT_SHAPE_FIELD_NAME not in config:
                    raise Exception
            except yaml.YAMLError:
                raise Exception


class YoloUltralyticsClassificationImporter(Importer):
    _FORMAT = YoloUltralyticsClassificationBase.NAME
    DETECT_CONFIDENCE = FormatDetectionConfidence.LOW

    @classmethod
    def find_sources(cls, path):
        if not osp.isdir(path):
            return []
        subfolders = [
            subfolder for name in os.listdir(path) if osp.isdir(subfolder := osp.join(path, name))
        ]
        if not subfolders:
            return []
        for subset_folder in subfolders:
            for name in os.listdir(subset_folder):
                if name in [YoloUltralyticsClassificationFormat.LABELS_FILE, DATASET_META_FILE]:
                    continue
                label_folder = osp.join(subset_folder, name)
                if not osp.isdir(label_folder) or not contains_only_images(label_folder):
                    return []

        return [{"url": path, "format": cls._FORMAT}]

    @property
    def can_stream(self) -> bool:
        return True
