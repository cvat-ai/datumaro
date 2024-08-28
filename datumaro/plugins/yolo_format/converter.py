# Copyright (C) 2019-2022 Intel Corporation
# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import math
import os
import os.path as osp
from collections import OrderedDict
from functools import cached_property
from itertools import cycle
from typing import Dict, List, Optional

import yaml

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    Points,
    PointsCategories,
    Polygon,
    Skeleton,
)
from datumaro.components.converter import Converter
from datumaro.components.dataset import DatasetPatch, ItemStatus
from datumaro.components.errors import DatasetExportError, MediaTypeError
from datumaro.components.extractor import DEFAULT_SUBSET_NAME, DatasetItem, IExtractor
from datumaro.components.media import Image
from datumaro.util import str_to_bool

from .format import YoloPath, YOLOv8Path


def _make_yolo_bbox(img_size, box):
    # https://github.com/pjreddie/darknet/blob/master/scripts/voc_label.py
    # <x> <y> <width> <height> - values relative to width and height of image
    # <x> <y> - are center of rectangle
    x = (box[0] + box[2]) / 2 / img_size[0]
    y = (box[1] + box[3]) / 2 / img_size[1]
    w = (box[2] - box[0]) / img_size[0]
    h = (box[3] - box[1]) / img_size[1]
    return x, y, w, h


def _bbox_annotation_as_polygon(bbox: Bbox) -> List[float]:
    points = bbox.as_polygon()

    def rotate_point(x: float, y: float):
        new_x = (
            center_x
            + math.cos(rotation_radians) * (x - center_x)
            - math.sin(rotation_radians) * (y - center_y)
        )
        new_y = (
            center_y
            + math.sin(rotation_radians) * (x - center_x)
            + math.cos(rotation_radians) * (y - center_y)
        )
        return new_x, new_y

    if rotation_radians := math.radians(bbox.attributes.get("rotation", 0)):
        center_x = bbox.x + bbox.w / 2
        center_y = bbox.y + bbox.h / 2
        points = [
            coordinate
            for x, y in zip(points[::2], points[1::2])
            for coordinate in rotate_point(x, y)
        ]
    return points


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

    def _save_annotation_file(self, annotation_path, yolo_annotation):
        with open(annotation_path, "w", encoding="utf-8") as f:
            f.write(yolo_annotation)

    def _export_item_annotation(self, item: DatasetItem, subset_dir: str) -> None:
        try:
            height, width = item.media.size

            yolo_annotation = ""

            for bbox in item.annotations:
                annotation_line = self._make_annotation_line(width, height, bbox)
                if annotation_line:
                    yolo_annotation += annotation_line

            annotation_path = osp.join(subset_dir, f"{item.id}{YoloPath.LABELS_EXT}")
            os.makedirs(osp.dirname(annotation_path), exist_ok=True)

            self._save_annotation_file(annotation_path, yolo_annotation)

        except Exception as e:
            self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))

    def _make_annotation_line(self, width: int, height: int, anno: Annotation) -> Optional[str]:
        if not isinstance(anno, Bbox) or anno.label is None:
            return
        if anno.attributes.get("rotation", 0) % 360.0 > 0.00001:
            return

        values = _make_yolo_bbox((width, height), anno.points)
        string_values = " ".join("%.6f" % p for p in values)
        return "%s %s\n" % (self._map_labels_for_save[anno.label], string_values)

    @cached_property
    def _labels_to_save(self) -> List[int]:
        return [
            label_id
            for label_id, label in enumerate(self._extractor.categories()[AnnotationType.label])
            if label.parent == ""
        ]

    @cached_property
    def _map_labels_for_save(self) -> Dict[int, int]:
        return {label_id: index for index, label_id in enumerate(self._labels_to_save)}

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


class YOLOv8DetectionConverter(YoloConverter):
    RESERVED_CONFIG_KEYS = YOLOv8Path.RESERVED_CONFIG_KEYS

    def __init__(
        self,
        extractor: IExtractor,
        save_dir: str,
        *,
        add_path_prefix: bool = True,
        config_file=None,
        **kwargs,
    ) -> None:
        super().__init__(extractor, save_dir, add_path_prefix=add_path_prefix, **kwargs)
        self._config_filename = config_file or YOLOv8Path.DEFAULT_CONFIG_FILE

    def _save_annotation_file(self, annotation_path, yolo_annotation):
        if yolo_annotation:
            super()._save_annotation_file(annotation_path, yolo_annotation)

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        parser = super().build_cmdline_parser(**kwargs)
        parser.add_argument(
            "--config-file",
            default=YOLOv8Path.DEFAULT_CONFIG_FILE,
            type=str,
            help="config file name (default: %(default)s)",
        )
        return parser

    def _save_config_files(self, subset_lists: Dict[str, str], **extra_config_fields):
        extractor = self._extractor
        save_dir = self._save_dir
        with open(osp.join(save_dir, self._config_filename), "w", encoding="utf-8") as f:
            label_categories = extractor.categories()[AnnotationType.label]
            data = dict(
                path=".",
                names={
                    index: label_categories[label_id].name
                    for label_id, index in self._map_labels_for_save.items()
                },
                **subset_lists,
                **extra_config_fields,
            )
            yaml.dump(data, f)

    @staticmethod
    def _make_image_subset_folder(save_dir: str, subset: str) -> str:
        return osp.join(save_dir, YOLOv8Path.IMAGES_FOLDER_NAME, subset)

    @staticmethod
    def _make_annotation_subset_folder(save_dir: str, subset: str) -> str:
        return osp.join(save_dir, YOLOv8Path.LABELS_FOLDER_NAME, subset)


class YOLOv8SegmentationConverter(YOLOv8DetectionConverter):
    def _make_annotation_line(self, width: int, height: int, anno: Annotation) -> Optional[str]:
        if anno.label is None or not isinstance(anno, Polygon):
            return
        values = [value / size for value, size in zip(anno.points, cycle((width, height)))]
        string_values = " ".join("%.6f" % p for p in values)
        return "%s %s\n" % (self._map_labels_for_save[anno.label], string_values)


class YOLOv8OrientedBoxesConverter(YOLOv8DetectionConverter):
    def _make_annotation_line(self, width: int, height: int, anno: Annotation) -> Optional[str]:
        if anno.label is None or not isinstance(anno, Bbox):
            return
        points = _bbox_annotation_as_polygon(anno)
        values = [value / size for value, size in zip(points, cycle((width, height)))]
        string_values = " ".join("%.6f" % p for p in values)
        return "%s %s\n" % (self._map_labels_for_save[anno.label], string_values)


class YOLOv8PoseConverter(YOLOv8DetectionConverter):
    @cached_property
    def _labels_to_save(self) -> List[int]:
        point_categories = self._extractor.categories().get(
            AnnotationType.points, PointsCategories.from_iterable([])
        )
        return sorted(point_categories.items)

    @cached_property
    def _max_number_of_points(self):
        point_categories = self._extractor.categories().get(AnnotationType.points)
        if point_categories is None or len(point_categories) == 0:
            return 0
        return max(len(category.labels) for category in point_categories.items.values())

    def _save_config_files(self, subset_lists: Dict[str, str], **extra_config_fields):
        super()._save_config_files(
            subset_lists=subset_lists,
            kpt_shape=[self._max_number_of_points, 3],
            **extra_config_fields,
        )

    def _make_annotation_line(self, width: int, height: int, skeleton: Annotation) -> Optional[str]:
        if skeleton.label is None or not isinstance(skeleton, Skeleton):
            return

        x, y, w, h = skeleton.get_bbox()
        bbox_values = _make_yolo_bbox((width, height), [x, y, x + w, y + h])
        bbox_string_values = " ".join("%.6f" % p for p in bbox_values)

        point_label_ids = [
            self._extractor.categories()[AnnotationType.label].find(
                name=child_label,
                parent=self._extractor.categories()[AnnotationType.label][skeleton.label].name,
            )[0]
            for child_label in self._extractor.categories()[AnnotationType.points]
            .items[skeleton.label]
            .labels
        ]

        points_values = [f"0.0 0.0 {Points.Visibility.absent.value}"] * self._max_number_of_points
        for element in skeleton.elements:
            assert len(element.points) == 2 and len(element.visibility) == 1
            position = point_label_ids.index(element.label)
            x = element.points[0] / width
            y = element.points[1] / height
            points_values[position] = f"{x:.6f} {y:.6f} {element.visibility[0].value}"

        return f"{self._map_labels_for_save[skeleton.label]} {bbox_string_values} {' '.join(points_values)}\n"
