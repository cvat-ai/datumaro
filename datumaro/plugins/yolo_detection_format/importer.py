# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os.path as osp
from glob import glob
import yaml
from collections import defaultdict, Counter
from typing import Any, Dict, List
from io import TextIOWrapper

from datumaro.components.errors import DatasetImportError
from datumaro.components.errors import DatasetNotFoundError
from datumaro.components.extractor import DEFAULT_SUBSET_NAME, Importer
from datumaro.components.format_detection import FormatDetectionConfidence, FormatDetectionContext
from datumaro.plugins.yolo_detection_format.extractor import YoloDetectionExtractor
from datumaro.util.os_util import extract_subset_name_from_parent

from .format import YoloDetectionPath


class YoloDetectionImporter(Importer):
    META_FILE = YoloDetectionPath.META_FILE

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        return parser

    @classmethod
    def detect(cls, context: FormatDetectionContext) -> FormatDetectionConfidence:
        context.require_file(cls.META_FILE)

        with context.require_any():
            with context.alternative():
                cls._check_ann_file(context.require_file("[Aa]nnotations/**/*.txt"), context)
            with context.alternative():
                cls._check_ann_file(context.require_file("[Ll]abels/**/*.txt"), context)

        return FormatDetectionConfidence.MEDIUM
    
    @classmethod
    def _check_ann_file(cls, fpath: str, context: FormatDetectionContext) -> None:
        with context.probe_text_file(
            fpath, "Requirements for the annotation file of yolo format"
        ) as fp:
            cls._check_ann_file_impl(fp)

    @classmethod
    def _check_ann_file_impl(cls, fp: TextIOWrapper) -> bool:
        for line in fp:
            fields = line.rstrip("\n").split(" ")
            if len(fields) != 5:
                raise DatasetImportError(
                    f"Yolo Detection format txt file should have 5 fields for each line, "
                    f"but the read line has {len(fields)} fields: fields={fields}."
                )

            for field in fields:
                if not field.replace(".", "").isdigit():
                    raise DatasetImportError(f"Each field should be a number but fields={fields}.")

            # Check the first line only
            return True

        raise DatasetImportError("Empty file is not allowed.")
    
    @classmethod
    def _find_loose(cls, path: str, dirname: str) -> List[Dict[str, Any]]:
        sources = cls._find_sources_recursive(
            path,
            ext=".txt",
            extractor_name="",
            dirname=dirname,
            filename="**/*",
            max_depth=1
        )
        if len(sources) == 0:
            return []

        subsets = defaultdict(list)

        for source in sources:
            subsets[extract_subset_name_from_parent(source["url"], path)].append(source["url"])

        sources = [
            {
                "url": osp.join(path),
                "format": "yolo_detection",
                "options": {
                    "subset": subset,
                    "urls": urls,
                },
            }
            for subset, urls in subsets.items()
        ]
        return sources

    @classmethod
    def find_sources(cls, path: str) -> List[Dict[str, Any]]:
        # Check obj.names first
        filename, ext = osp.splitext(cls.META_FILE)
        obj_names_files = cls._find_sources_recursive(
            path,
            ext=ext,
            extractor_name="",
            dirname="",
            filename=filename,
            max_depth=1
        )
        if len(obj_names_files) == 0:
            return []

        sources = []

        for obj_names_file in obj_names_files:
            base_path = osp.dirname(obj_names_file["url"])
            if found := cls._find_loose(base_path, "[Aa]nnotations"):
                sources += found

            if found := cls._find_loose(base_path, "[Ll]abels"):
                sources += found

        return sources

    def __call__(self, path, **extra_params):
        subsets = self.find_sources(path)
        return subsets