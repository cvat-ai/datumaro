# Copyright (C) 2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import logging as log
import os
import os.path as osp
from collections import OrderedDict, defaultdict

import yaml

from datumaro.components.annotation import AnnotationType, Polygon
from datumaro.components.converter import Converter
from datumaro.components.dataset import ItemStatus
from datumaro.components.errors import DatasetExportError, MediaTypeError, DatumaroError
from datumaro.components.extractor import DEFAULT_SUBSET_NAME, DatasetItem, IExtractor
from datumaro.components.media import Image
from datumaro.util import str_to_bool

from .format import YoloSegmentationPath


class YoloSegmentationConverter(Converter):
    DEFAULT_IMAGE_EXT = ".jpg"

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
        

    def _check_dataset(self):
        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        subset_names = set(self._extractor.subsets().keys())

        for subset in subset_names:
            if subset not in YoloSegmentationPath.ALLOWED_SUBSET_NAMES:
                raise DatasetExportError(
                    f"The allowed subset name is in {YoloSegmentationPath.ALLOWED_SUBSET_NAMES}, "
                    f'so that subset "{subset}" is not allowed.'
                )

        for must_name in YoloSegmentationPath.MUST_SUBSET_NAMES:
            if must_name not in subset_names:
                raise DatasetExportError(
                    f'Subset "{must_name}" is not in {subset_names}, '
                    "but Yolov8 requires both of them."
                )
            
    def _export_media(self, item: DatasetItem, subset_img_dir: str) -> str:
        try:
            if not item.media or not (item.media.has_data or item.media.has_size):
                raise DatasetExportError(
                    "Failed to export item '%s': " "item has no image info" % item.id
                )

            image_name = self._make_image_filename(item)
            image_fpath = osp.join(subset_img_dir, image_name)

            if self._save_media:
                self._save_image(item, image_fpath)

            return image_fpath

        except Exception as e:
            self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))

    def apply(self):
        extractor = self._extractor
        save_dir = self._save_dir

        if self._extractor.media_type() and not issubclass(self._extractor.media_type(), Image):
            raise MediaTypeError("Media type is not an image")

        os.makedirs(save_dir, exist_ok=True)

        try:
            self._check_dataset()
        except DatumaroError as e:
            self._ctx.error_policy.fail(e)

        if self._save_dataset_meta:
            self._save_meta_file(self._save_dir)

        yaml_dict = {}

        subsets = self._extractor.subsets()
        pbars = self._ctx.progress_reporter.split(len(subsets))

        image_fpaths = defaultdict(list)

        for (subset_name, subset), pbar in zip(subsets.items(), pbars):
            if subset_name in YoloSegmentationPath.RESERVED_CONFIG_KEYS:
                raise DatasetExportError(
                    f"Can't export '{subset_name}' subset in Yolov8 format, this word is reserved."
                )

            subset_fpath = osp.join(save_dir, subset_name + ".txt")

            subset_img_dir = osp.join(save_dir, "images", subset_name)
            os.makedirs(subset_img_dir, exist_ok=True)

            subset_label_dir = osp.join(save_dir, "labels", subset_name)
            os.makedirs(subset_label_dir, exist_ok=True)

            yaml_dict[subset_name] = subset_fpath

            image_paths = OrderedDict()

            for item in pbar.iter(subset, desc=f"Exporting '{subset_name}'"):
                image_fpath = self._export_media(item, subset_img_dir)
                self._export_item_annotation(item, subset_label_dir)

                image_fpaths[subset_name].append(osp.relpath(image_fpath, save_dir))
        
        for subset_name, img_fpath_list in image_fpaths.items():
            subset_fname = subset_name + ".txt"
            with open(osp.join(save_dir, subset_fname), "w") as fp:
                # Prefix (os.curdir + os.sep) is required by Ultralytics
                # Please see https://github.com/ultralytics/ultralytics/blob/30fc4b537ff1d9b115bc1558884f6bc2696a282c/ultralytics/yolo/data/utils.py#L40-L43
                fp.writelines(
                    [os.curdir + os.sep + img_fpath + "\n" for img_fpath in img_fpath_list]
                )
            yaml_dict[subset_name] = subset_fname

        label_categories = extractor.categories()[AnnotationType.label]
        label_ids = {idx: label.name for idx, label in enumerate(label_categories.items)}
        yaml_dict["names"] = label_ids

        with open(osp.join(save_dir, "data.yaml"), "w") as fp:
            yaml.safe_dump(yaml_dict, fp, sort_keys=False, allow_unicode=True)

    def _export_item_annotation(self, item, subset_dir):
        try:
            height, width = item.media.size

            yolo_annotation = ""

            for mask in item.annotations:
                if not isinstance(mask, Polygon) or mask.label is None:
                    continue

                for i in range(0, len(mask.points), 2):
                    mask.points[i] = mask.points[i] / width
                    mask.points[i + 1] = mask.points[i + 1] / height

                yolo_mask = " ".join("%.6f" % p for p in mask.points)
                yolo_annotation += "%s %s\n" % (mask.label, yolo_mask)

            annotation_path = osp.join(subset_dir, "%s.txt" % item.id)
            os.makedirs(osp.dirname(annotation_path), exist_ok=True)

            with open(annotation_path, "w", encoding="utf-8") as f:
                f.write(yolo_annotation)

        except Exception as e:
            self._ctx.error_policy.report_item_error(e, item_id=(item.id, item.subset))

    @classmethod
    def patch(cls, dataset, patch, save_dir, **kwargs):
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
                subset = YoloSegmentationPath.DEFAULT_SUBSET_NAME
            subset_dir = osp.join(save_dir, "obj_%s_data" % subset)

            image_path = osp.join(subset_dir, conv._make_image_filename(item))
            if osp.isfile(image_path):
                os.remove(image_path)

            ann_path = osp.join(subset_dir, "%s.txt" % item.id)
            if osp.isfile(ann_path):
                os.remove(ann_path)
