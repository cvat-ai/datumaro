# Copyright (C) 2019-2021 Intel Corporation
#
# SPDX-License-Identifier: MIT


class YoloPath:
    DEFAULT_SUBSET_NAME = "train"
    SUBSET_NAMES = ["train", "valid"]
    RESERVED_CONFIG_KEYS = ["backup", "classes", "names"]
    LABELS_EXT = ".txt"
    SUBSET_LIST_EXT = ".txt"


class Yolo8Path(YoloPath):
    CONFIG_FILE_EXT = ".yaml"
    DEFAULT_CONFIG_FILE = "data.yaml"
    RESERVED_CONFIG_KEYS = YoloPath.RESERVED_CONFIG_KEYS + [
        "path",
        "kpt_shape",
        "flip_idx",
    ]
    IMAGES_FOLDER_NAME = "images"
    LABELS_FOLDER_NAME = "labels"


class Yolo8PoseFormat:
    KPT_SHAPE_FIELD_NAME = "kpt_shape"
