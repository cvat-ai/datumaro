# Copyright (C) 2019-2022 Intel Corporation
# Copyright (C) 2023 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import os.path as osp
import re
from collections import OrderedDict
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import yaml

from datumaro.components.annotation import (
    Annotation,
    AnnotationType,
    Bbox,
    LabelCategories,
    Polygon,
)
from datumaro.components.errors import (
    DatasetImportError,
    InvalidAnnotationError,
    UndeclaredLabelError,
)
from datumaro.components.extractor import DatasetItem, Extractor, SourceExtractor
from datumaro.components.media import Image
from datumaro.util.image import (
    DEFAULT_IMAGE_META_FILE_NAME,
    ImageMeta,
    load_image,
    load_image_meta_file,
)
from datumaro.util.meta_file_util import has_meta_file, parse_meta_file
from datumaro.util.os_util import split_path

from .format import Yolo8Path, YoloPath

T = TypeVar("T")


class YoloExtractor(SourceExtractor):
    RESERVED_CONFIG_KEYS = YoloPath.RESERVED_CONFIG_KEYS

    class Subset(Extractor):
        def __init__(self, name: str, parent: YoloExtractor):
            super().__init__()
            self._name = name
            self._parent = parent
            self.items: Dict[str, Union[str, DatasetItem]] = OrderedDict()

        def __iter__(self):
            for item_id in self.items:
                item = self._parent._get(item_id, self._name)
                if item is not None:
                    yield item

        def __len__(self):
            return len(self.items)

        def categories(self):
            return self._parent.categories()

    def __init__(
        self,
        config_path: str,
        image_info: Union[None, str, ImageMeta] = None,
        **kwargs,
    ) -> None:
        if not osp.isfile(config_path):
            raise DatasetImportError(f"Can't read dataset descriptor file '{config_path}'")

        super().__init__(**kwargs)

        rootpath = osp.dirname(config_path)
        self._path = rootpath

        assert image_info is None or isinstance(image_info, (str, dict))
        if image_info is None:
            image_info = osp.join(rootpath, DEFAULT_IMAGE_META_FILE_NAME)
            if not osp.isfile(image_info):
                image_info = {}
        if isinstance(image_info, str):
            image_info = load_image_meta_file(image_info)

        self._image_info = image_info

        config = self._parse_config(config_path)

        self._categories = {AnnotationType.label: self._load_categories(config)}

        # The original format is like this:
        #
        # classes = 2
        # train  = data/train.txt
        # valid  = data/test.txt
        # names = data/obj.names
        # backup = backup/
        #
        # To support more subset names, we disallow subsets
        # called 'classes', 'names' and 'backup'.
        subsets = {k: v for k, v in config.items() if k not in self.RESERVED_CONFIG_KEYS}

        for subset_name, list_path in subsets.items():
            subset = YoloExtractor.Subset(subset_name, self)
            subset.items = OrderedDict(
                (self.name_from_path(p), self.localize_path(p))
                for p in self._iterate_over_image_paths(subset_name, list_path)
            )
            subsets[subset_name] = subset

        self._subsets: Dict[str, YoloExtractor.Subset] = subsets

    def _iterate_over_image_paths(self, subset_name: str, list_path: str):
        list_path = osp.join(self._path, self.localize_path(list_path))
        if not osp.isfile(list_path):
            raise InvalidAnnotationError(f"Can't find '{subset_name}' subset list file")

        with open(list_path, "r", encoding="utf-8") as f:
            yield from (p for p in f if p.strip())

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

    @staticmethod
    def localize_path(path: str) -> str:
        """
        Removes the "data/" prefix from the path
        """

        path = osp.normpath(path.strip()).replace("\\", "/")
        default_base = "data/"
        if path.startswith(default_base):
            path = path[len(default_base) :]
        return path

    @classmethod
    def name_from_path(cls, path: str) -> str:
        """
        Obtains <image name> from the path like [data/]<subset>_obj/<image_name>.ext

        <image name> can be <a/b/c/filename>, so it is
        more involved than just calling "basename()".
        """

        path = cls.localize_path(path)

        parts = split_path(path)
        if 1 < len(parts) and not osp.isabs(path):
            path = osp.join(*parts[1:])  # pylint: disable=no-value-for-parameter

        return osp.splitext(path)[0]

    @classmethod
    def _image_loader(cls, *args, **kwargs):
        return load_image(*args, **kwargs, keep_exif=True)

    def _get_labels_path_from_image_path(self, image_path: str) -> str:
        return osp.splitext(image_path)[0] + YoloPath.LABELS_EXT

    def _get(self, item_id: str, subset_name: str) -> Optional[DatasetItem]:
        subset = self._subsets[subset_name]
        item = subset.items[item_id]

        if isinstance(item, str):
            try:
                image_size = self._image_info.get(item_id)
                image_path = osp.join(self._path, item)

                if image_size:
                    image = Image(path=image_path, size=image_size)
                else:
                    image = Image(path=image_path, data=self._image_loader)

                anno_path = self._get_labels_path_from_image_path(image.path)
                annotations = self._parse_annotations(
                    anno_path, image, item_id=(item_id, subset_name)
                )

                item = DatasetItem(
                    id=item_id, subset=subset_name, media=image, annotations=annotations
                )
                subset.items[item_id] = item
            except Exception as e:
                self._ctx.error_policy.report_item_error(e, item_id=(item_id, subset_name))
                subset.items.pop(item_id)
                item = None

        return item

    @staticmethod
    def _parse_field(value: str, cls: Type[T], field_name: str) -> T:
        try:
            return cls(value)
        except Exception as e:
            raise InvalidAnnotationError(
                f"Can't parse {field_name} from '{value}'. Expected {cls}"
            ) from e

    def _parse_annotations(
        self, anno_path: str, image: Image, *, item_id: Tuple[str, str]
    ) -> List[Annotation]:
        lines = []
        with open(anno_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(line)

        annotations = []

        if lines:
            # Use image info as late as possible to avoid unnecessary image loading
            if image.size is None:
                raise DatasetImportError(
                    f"Can't find image info for '{self.localize_path(image.path)}'"
                )
            image_height, image_width = image.size

        for line in lines:
            try:
                annotations.append(
                    self._load_one_annotation(line.split(), image_height, image_width)
                )
            except Exception as e:
                self._ctx.error_policy.report_annotation_error(e, item_id=item_id)

        return annotations

    def _load_one_annotation(
        self, parts: List[str], image_height: int, image_width: int
    ) -> Annotation:
        if len(parts) != 5:
            raise InvalidAnnotationError(
                f"Unexpected field count {len(parts)} in the bbox description. "
                "Expected 5 fields (label, xc, yc, w, h)."
            )
        label_id, xc, yc, w, h = parts

        label_id = self._parse_field(label_id, int, "bbox label id")
        if label_id not in self._categories[AnnotationType.label]:
            raise UndeclaredLabelError(str(label_id))

        w = self._parse_field(w, float, "bbox width")
        h = self._parse_field(h, float, "bbox height")
        x = self._parse_field(xc, float, "bbox center x") - w * 0.5
        y = self._parse_field(yc, float, "bbox center y") - h * 0.5

        return Bbox(
            x * image_width,
            y * image_height,
            w * image_width,
            h * image_height,
            label=label_id,
        )

    def _load_categories(self, config: Dict[str, str]):
        names_path = config.get("names")
        if not names_path:
            raise InvalidAnnotationError(f"Failed to parse names file path from config")

        names_path = osp.join(self._path, self.localize_path(names_path))

        if has_meta_file(osp.dirname(names_path)):
            return LabelCategories.from_iterable(parse_meta_file(osp.dirname(names_path)).keys())

        label_categories = LabelCategories()

        with open(names_path, "r", encoding="utf-8") as f:
            for label in f:
                label = label.strip()
                if label:
                    label_categories.add(label)

        return label_categories

    def __iter__(self):
        subsets = self._subsets
        pbars = self._ctx.progress_reporter.split(len(subsets))
        for pbar, (subset_name, subset) in zip(pbars, subsets.items()):
            for item in pbar.iter(subset, desc=f"Parsing '{subset_name}'"):
                yield item

    def __len__(self):
        return sum(len(s) for s in self._subsets.values())

    def get_subset(self, name):
        return self._subsets[name]


class Yolo8Extractor(YoloExtractor):
    RESERVED_CONFIG_KEYS = Yolo8Path.RESERVED_CONFIG_KEYS

    @staticmethod
    def _parse_config(path: str) -> Dict[str, Any]:
        with open(path) as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError:
                raise InvalidAnnotationError("Failed to parse config file")

    def _load_categories(self, config: Dict[str, str]) -> LabelCategories:
        names = config.get("names")

        if names is not None:
            if isinstance(names, dict):
                return LabelCategories.from_iterable([names[i] for i in range(len(names))])
            if isinstance(names, list):
                return LabelCategories.from_iterable(names)

        raise InvalidAnnotationError(f"Failed to parse names from config")

    def _get_labels_path_from_image_path(self, image_path: str) -> str:
        relative_image_path = osp.relpath(
            image_path, osp.join(self._path, Yolo8Path.IMAGES_FOLDER_NAME)
        )
        relative_labels_path = osp.splitext(relative_image_path)[0] + Yolo8Path.LABELS_EXT
        return osp.join(self._path, Yolo8Path.LABELS_FOLDER_NAME, relative_labels_path)

    @classmethod
    def name_from_path(cls, path: str) -> str:
        """
        Obtains <image name> from the path like [data/]images/<subset>/<image_name>.ext

        <image name> can be <a/b/c/filename>, so it is
        more involved than just calling "basename()".
        """
        path = cls.localize_path(path)

        parts = split_path(path)
        if 2 < len(parts) and not osp.isabs(path):
            path = osp.join(*parts[2:])  # pylint: disable=no-value-for-parameter
        return osp.splitext(path)[0]

    def _iterate_over_image_paths(
        self, subset_name: str, subset_images_source: Union[str, List[str]]
    ):
        if isinstance(subset_images_source, str):
            if subset_images_source.endswith(YoloPath.SUBSET_LIST_EXT):
                yield from super()._iterate_over_image_paths(subset_name, subset_images_source)
            else:
                path = osp.join(self._path, self.localize_path(subset_images_source))
                if not osp.isdir(path):
                    raise InvalidAnnotationError(f"Can't find '{subset_name}' subset image folder")
                yield from (
                    osp.relpath(osp.join(root, file), self._path)
                    for root, dirs, files in os.walk(path)
                    for file in files
                    if osp.isfile(osp.join(root, file))
                )
        else:
            yield from subset_images_source


class Yolo8SegmentationExtractor(Yolo8Extractor):
    def _load_segmentation_annotation(
        self, parts: List[str], image_height: int, image_width: int
    ) -> Polygon:
        label_id = self._parse_field(parts[0], int, "Polygon label id")
        if label_id not in self._categories[AnnotationType.label]:
            raise UndeclaredLabelError(str(label_id))

        points = [
            self._parse_field(
                value, float, f"polygon point {idx // 2} {'x' if idx % 2 == 0 else 'y'}"
            )
            for idx, value in enumerate(parts[1:])
        ]
        scaled_points = [
            value * size for value, size in zip(points, cycle((image_width, image_height)))
        ]
        return Polygon(scaled_points, label=label_id)

    def _load_one_annotation(
        self, parts: List[str], image_height: int, image_width: int
    ) -> Annotation:
        if len(parts) > 5 and len(parts) % 2 == 1:
            return self._load_segmentation_annotation(parts, image_height, image_width)
        raise InvalidAnnotationError(
            f"Unexpected field count {len(parts)} in the polygon description. "
            "Expected odd number > 5 of fields for segment annotation (label, x1, y1, x2, y2, x3, y3, ...)"
        )
