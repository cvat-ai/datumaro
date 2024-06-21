# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto
from typing import Dict
import re
import yaml

class YoloDetectionPath:
    DEFAULT_SUBSET_NAME = "train"
    MUST_SUBSET_NAMES = ["train"]
    ALLOWED_SUBSET_NAMES = ["train", "valid", "test", "val"]
    RESERVED_CONFIG_KEYS = ["backup", "classes", "names"]
    META_FILE = "data.yaml"

    @staticmethod
    def _parse_config(path: str) -> Dict[str, str]:
        with open(path, "r") as fp:
            loaded = yaml.safe_load(fp.read())

        if not isinstance(loaded, dict):
            raise Exception("Invalid config format")
        
        return loaded
