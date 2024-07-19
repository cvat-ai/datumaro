# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import Any, Dict, List

from datumaro import Importer
from datumaro.components.format_detection import FormatDetectionContext


class _YoloImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        context.require_file("obj.data")

    @classmethod
    def find_sources(cls, path) -> List[Dict[str, Any]]:
        return cls._find_sources_recursive(path, ".data", "yolo")


class YoloImporter(Importer):
    SUB_IMPORTERS: List[Importer] = [
        _YoloImporter,
    ]

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        with context.require_any():
            for importer_cls in cls.SUB_IMPORTERS:
                with context.alternative():
                    return importer_cls.detect(context)

        context.fail("Any yolo format is not detected.")

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        for importer_cls in cls.SUB_IMPORTERS:
            if sources := importer_cls.find_sources(path):
                return sources

        return []
