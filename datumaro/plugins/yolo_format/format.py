# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT


class YoloPath:
    DEFAULT_SUBSET_NAME = "train"
    SUBSET_NAMES = ["train", "valid"]
    RESERVED_CONFIG_KEYS = ["backup", "classes", "names"]
    LABELS_EXT = ".txt"


class Yolo8Path(YoloPath):
    RESERVED_CONFIG_KEYS = YoloPath.RESERVED_CONFIG_KEYS + [
        "path",
    ]
    IMAGES_FOLDER_NAME = "images"
    LABELS_FOLDER_NAME = "labels"
