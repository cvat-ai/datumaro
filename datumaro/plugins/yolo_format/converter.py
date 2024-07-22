# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from collections import OrderedDict
from typing import Dict

import yaml

from datumaro.components.annotation import AnnotationType, Bbox
from datumaro.components.converter import Converter
from datumaro.components.dataset import DatasetPatch, ItemStatus
from datumaro.components.errors import DatasetExportError, MediaTypeError
from datumaro.components.extractor import DEFAULT_SUBSET_NAME, DatasetItem, IExtractor
from datumaro.components.media import Image
from datumaro.util import str_to_bool

from .format import Yolo8Path, YoloPath


def _make_yolo_bbox(img_size, box):
    # https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
    # <x> <y> <width> <height> - values relative to width and height of image
    # <x> <y> - are center of rectangle
    x = (box[0] + box[2]) / 2 / img_size[0]
    y = (box[1] + box[3]) / 2 / img_size[1]
    w = (box[2] - box[0]) / img_size[0]
    h = (box[3] - box[1]) / img_size[1]
    return x, y, w, h


class YoloConverter(Converter):
    # https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
    DEFAULT_IMAGE_EXT = ".jpg"
    RESERVED_CONFIG_KEYS = YoloPath.RESERVED_CONFIG_KEYS

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--add-path-prefix",
            default=True,
            type=str_to_bool,
            help="Add the 'data/' prefix for paths in the dataset info (default: %(default)s)",
        )
        return parser

    def __init__(
        self, extractor: IExtractor, save_dir: str, *, add_path_prefix: bool = True, **kwargs
    ) -> None:
        super().__init__(extractor, save_dir, **kwargs)

        self._prefix = "data" if add_path_prefix else ""

    def apply(self):
        save_dir = self._save_dir

        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(save_dir, exist_ok=True)

        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        subset_lists = OrderedDict()

        subsets = self._extractor.subsets()
        pbars = self._ctx.progress_reporter.split(len(subsets))
        for (subset_name, subset), pbar in zip(subsets.items(), pbars):
            if not subset_name or subset_name == DEFAULT_SUBSET_NAME:
                subset_name = YoloPath.DEFAULT_SUBSET_NAME
            elif subset_name in self.RESERVED_CONFIG_KEYS:
                raise DatasetExportError(
                    f"Can't export '{subset_name}' subset in YOLO format, this word is reserved."
                )

            subset_image_dir = self._make_image_subset_folder(save_dir, subset_name)
            subset_anno_dir = self._make_annotation_subset_folder(save_dir, subset_name)
            os.makedirs(subset_image_dir, exist_ok=True)
            os.makedirs(subset_anno_dir, exist_ok=True)

            image_paths = OrderedDict()
            for item in pbar.iter(subset, desc=f"Exporting '{subset_name}'"):
                try:
                    image_fpath = self._export_media(item, subset_image_dir)
                    image_name = osp.relpath(image_fpath, subset_image_dir)
                    image_paths[item.id] = osp.join(
                        self._prefix, osp.relpath(subset_image_dir, save_dir), image_name
                    )

                    self._export_item_annotation(item, subset_anno_dir)

                except Exception as e:
                    self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))

            if subset_list_name := self._make_subset_list_file(subset_name, image_paths):
                subset_lists[subset_name] = subset_list_name

        self._save_config_files(subset_lists)

    def _save_config_files(self, subset_lists: Dict[str, str]):
        extractor = self._extractor
        save_dir = self._save_dir
        label_categories = extractor.categories()[AnnotationType.label]
        label_ids = {label.name: idx for idx, label in enumerate(label_categories.items)}
        with open(osp.join(save_dir, "obj.names"), "w", encoding="utf-8") as f:
            f.writelines("%s\n" % l[0] for l in sorted(label_ids.items(), key=lambda x: x[1]))

        with open(osp.join(save_dir, "obj.data"), "w", encoding="utf-8") as f:
            f.write(f"classes = {len(label_ids)}\n")

            for subset_name, subset_list_name in subset_lists.items():
                f.write(
                    "%s = %s\n"
                    % (subset_name, osp.join(self._prefix, subset_list_name).replace("\\", "/"))
                )

            f.write("names = %s\n" % osp.join(self._prefix, "obj.names"))
            f.write("backup = backup/\n")

    def _make_subset_list_file(self, subset_name, image_paths):
        subset_list_name = f"{subset_name}{YoloPath.SUBSET_LIST_EXT}"
        subset_list_path = osp.join(self._save_dir, subset_list_name)
        if self._patch and subset_name in self._patch.updated_subsets and not image_paths:
            if osp.isfile(subset_list_path):
                os.remove(subset_list_path)
            return

        with open(subset_list_path, "w", encoding="utf-8") as f:
            f.writelines("%s\n" % s.replace("\\", "/") for s in image_paths.values())
        return subset_list_name

    def _export_media(self, item: DatasetItem, subset_img_dir: str) -> str:
        try:
            if not item.media or not (item.media.has_data or item.media.has_size):
                raise DatasetExportError(
                    "Failed to export item '%s': " "item has no image info" % item.id
                )

            image_name = self._make_image_filename(item)
            image_fpath = osp.join(subset_img_dir, image_name)

            if self._save_media:
                if item.media:
                    self._save_image(item, image_fpath)
                else:
                    log.warning("Item '%s' has no image" % item.id)

            return image_fpath

        except Exception as e:
            self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))

    def _export_item_annotation(self, item: DatasetItem, subset_dir: str) -> None:
        try:
            height, width = item.media.size

            yolo_annotation = ""

            for bbox in item.annotations:
                if not isinstance(bbox, Bbox) or bbox.label is None:
                    continue

                yolo_bb = _make_yolo_bbox((width, height), bbox.points)
                yolo_bb = " ".join("%.6f" % p for p in yolo_bb)
                yolo_annotation += "%s %s\n" % (bbox.label, yolo_bb)

            annotation_path = osp.join(subset_dir, f"{item.id}{YoloPath.LABELS_EXT}")
            os.makedirs(osp.dirname(annotation_path), exist_ok=True)

            with open(annotation_path, "w", encoding="utf-8") as f:
                f.write(yolo_annotation)

        except Exception as e:
            self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))

    @staticmethod
    def _make_image_subset_folder(save_dir: str, subset: str) -> str:
        return osp.join(save_dir, f"obj_{subset}_data")

    @staticmethod
    def _make_annotation_subset_folder(save_dir: str, subset: str) -> str:
        return osp.join(save_dir, f"obj_{subset}_data")

    @classmethod
    def patch(cls, dataset: IExtractor, patch: DatasetPatch, save_dir: str, **kwargs):
        conv = cls(dataset, save_dir=save_dir, **kwargs)
        conv._patch = patch
        conv.apply()

        for (item_id, subset), status in patch.updated_items.items():
            if status != ItemStatus.removed:
                item = patch.data.get(item_id, subset)
            else:
                item = DatasetItem(item_id, subset=subset)

            if not (status == ItemStatus.removed or not item.media):
                continue

            if subset == DEFAULT_SUBSET_NAME:
                subset = YoloPath.DEFAULT_SUBSET_NAME
            subset_image_dir = cls._make_image_subset_folder(save_dir, subset)
            subset_anno_dir = cls._make_annotation_subset_folder(save_dir, subset)

            image_path = osp.join(subset_image_dir, conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.remove(image_path)

            ann_path = osp.join(subset_anno_dir, f"{item.id}{YoloPath.LABELS_EXT}")
            if osp.isfile(ann_path):
                os.remove(ann_path)


class Yolo8Converter(YoloConverter):
    RESERVED_CONFIG_KEYS = Yolo8Path.RESERVED_CONFIG_KEYS

    def _save_config_files(self, subset_lists: Dict[str, str]):
        extractor = self._extractor
        save_dir = self._save_dir
        with open(osp.join(save_dir, "data.yaml"), "w", encoding="utf-8") as f:
            label_categories = extractor.categories()[AnnotationType.label]
            data = dict(
                path=".",
                names={idx: label.name for idx, label in enumerate(label_categories.items)},
                **subset_lists,
            )
            yaml.dump(data, f)

    @staticmethod
    def _make_image_subset_folder(save_dir: str, subset: str) -> str:
        return osp.join(save_dir, Yolo8Path.IMAGES_FOLDER_NAME, subset)

    @staticmethod
    def _make_annotation_subset_folder(save_dir: str, subset: str) -> str:
        return osp.join(save_dir, Yolo8Path.LABELS_FOLDER_NAME, subset)
