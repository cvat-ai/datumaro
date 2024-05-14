# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from unittest import mock
from typing import Optional, Callable

from datumaro.components.errors import DatasetNotFoundError

def wrap_find_sources_recursive(importer):
    @classmethod
    def updated_find_sources_recursive(
        cls,
        path: str,
        ext: Optional[str],
        extractor_name: str,
        filename: str = "*",
        dirname: str = "",
        file_filter: Optional[Callable[[str], bool]] = None,
        max_depth: int = 3,
    ):
        sources = super(importer, cls)._find_sources_recursive(
            path, ext, extractor_name,
            filename, dirname, file_filter, max_depth
        )
        if not sources:
            cls._not_found_error_data = {"ext": ext}

    return updated_find_sources_recursive

def wrap_generate_not_found_error(importer):
    @classmethod
    def updated_generate_not_found_error(cls, path):
        return DatasetNotFoundError(path, cls._not_found_error_data.get("ext"))

    return updated_generate_not_found_error

def wrap_importer(importer):
    mock.patch.object(importer, '_find_sources_recursive', new=wrap_find_sources_recursive(importer)).start()
    mock.patch.object(importer, '_generate_not_found_error', new=wrap_generate_not_found_error(importer)).start()
    mock.patch.object(importer, '_not_found_error_data', new={"ext": ""}, create=True).start()
