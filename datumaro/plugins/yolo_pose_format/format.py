# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto
from typing import Dict
import re


class YoloPosePath:
    DEFAULT_SUBSET_NAME = "train"
    MUST_SUBSET_NAMES = ["train"]
    ALLOWED_SUBSET_NAMES = ["train", "valid", "test", "val"]
    RESERVED_CONFIG_KEYS = ["backup", "classes", "names"]
    META_FILE = "data.yaml"

    @staticmethod
    def _parse_config(path: str) -> Dict[str, str]:
        with open(path, "r", encoding="utf-8") as f:
            config_lines = f.readlines()

        config = {}

        for line in config_lines:
            match = re.match(r"^\s*(\w+)\s*=\s*(.+)$", line)
            if not match:
                continue

            key = match.group(1)
            value = match.group(2)
            config[key] = value

        return config
