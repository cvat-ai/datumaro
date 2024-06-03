# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT

from enum import Enum, auto
from typing import Dict
import re

# class Yolov8Task(Enum):
#     instances = auto()
#     person_keypoints = auto()
#     captions = auto()
#     labels = auto()  # extension, does not exist in the original Yolov8 format
#     image_info = auto()
#     panoptic = auto()
#     stuff = auto()


class Yolov8Path:
    DEFAULT_SUBSET_NAME = "train"
    MUST_SUBSET_NAMES = ["train", "valid"]
    ALLOWED_SUBSET_NAMES = ["train", "valid", "test"]
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
