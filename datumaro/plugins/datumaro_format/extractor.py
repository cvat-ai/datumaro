# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from typing import List, Union

from datumaro.components.annotation import (
    AnnotationType,
    Bbox,
    Caption,
    Cuboid3d,
    Label,
    LabelCategories,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RleMask,
    Skeleton,
)
from datumaro.components.errors import DatasetImportError, InvalidAnnotationError
from datumaro.components.extractor import DatasetItem, Importer, SourceExtractor
from datumaro.components.format_detection import FormatDetectionContext
from datumaro.components.media import Image, MediaElement, PointCloud
from datumaro.util import parse_json, parse_json_file, take_by

from .format import DatumaroPath


class DatumaroExtractor(SourceExtractor):
    def __init__(self, path):
        assert osp.isfile(path), path

        rootpath = ""
        if path.endswith(osp.join(DatumaroPath.ANNOTATIONS_DIR, osp.basename(path))):
            rootpath = path.rsplit(DatumaroPath.ANNOTATIONS_DIR, maxsplit=1)[0]

        images_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.IMAGES_DIR)):
            images_dir = osp.join(rootpath, DatumaroPath.IMAGES_DIR)
        self._images_dir = images_dir

        pcd_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.PCD_DIR)):
            pcd_dir = osp.join(rootpath, DatumaroPath.PCD_DIR)
        self._pcd_dir = pcd_dir

        related_images_dir = ""
        if rootpath and osp.isdir(osp.join(rootpath, DatumaroPath.RELATED_IMAGES_DIR)):
            related_images_dir = osp.join(rootpath, DatumaroPath.RELATED_IMAGES_DIR)
        self._related_images_dir = related_images_dir

        super().__init__(subset=osp.splitext(osp.basename(path))[0])

        parsed_anns = parse_json_file(path)
        self._categories = self._load_categories(parsed_anns)
        self._items = self._load_items(parsed_anns)

    @staticmethod
    def _load_categories(parsed):
        categories = {}

        parsed_label_cat = parsed["categories"].get(AnnotationType.label.name)
        if parsed_label_cat:
            label_categories = LabelCategories(attributes=parsed_label_cat.get("attributes", []))
            for item in parsed_label_cat["labels"]:
                label_categories.add(
                    item["name"], parent=item["parent"], attributes=item.get("attributes", [])
                )

            categories[AnnotationType.label] = label_categories

        parsed_mask_cat = parsed["categories"].get(AnnotationType.mask.name)
        if parsed_mask_cat:
            colormap = {}
            for item in parsed_mask_cat["colormap"]:
                colormap[int(item["label_id"])] = (item["r"], item["g"], item["b"])

            mask_categories = MaskCategories(colormap=colormap)
            categories[AnnotationType.mask] = mask_categories

        parsed_points_cat = parsed["categories"].get(AnnotationType.points.name)
        if parsed_points_cat:
            point_categories = PointsCategories()
            for item in parsed_points_cat["items"]:
                point_categories.add(int(item["label_id"]), item["labels"], joints=item["joints"])

            categories[AnnotationType.points] = point_categories

        return categories

    def _load_items(self, parsed):
        items = []
        for item_desc in parsed["items"]:
            item_id = item_desc["id"]

            media = None
            image_info = item_desc.get("image")
            if image_info:
                image_filename = image_info.get("path") or item_id + DatumaroPath.IMAGE_EXT
                image_path = osp.join(self._images_dir, self._subset, image_filename)
                if not osp.isfile(image_path):
                    # backward compatibility
                    old_image_path = osp.join(self._images_dir, image_filename)
                    if osp.isfile(old_image_path):
                        image_path = old_image_path

                media = Image(path=image_path, size=image_info.get("size"))
                self._media_type = Image

            pcd_info = item_desc.get("point_cloud")
            if media and pcd_info:
                raise DatasetImportError("Dataset cannot contain multiple media types")
            if pcd_info:
                pcd_path = pcd_info.get("path")
                point_cloud = osp.join(self._pcd_dir, self._subset, pcd_path)

                related_images = None
                ri_info = item_desc.get("related_images")
                if ri_info:
                    related_images = [
                        Image(
                            size=ri.get("size"),
                            path=osp.join(
                                self._related_images_dir, self._subset, item_id, ri.get("path")
                            ),
                        )
                        for ri in ri_info
                    ]

                media = PointCloud(point_cloud, extra_images=related_images)
                self._media_type = PointCloud

            media_desc = item_desc.get("media")
            if not media and media_desc and media_desc.get("path"):
                media = MediaElement(path=media_desc.get("path"))
                self._media_type = MediaElement

            annotations = self._load_annotations(item_desc)

            item = DatasetItem(
                id=item_id,
                subset=self._subset,
                annotations=annotations,
                media=media,
                attributes=item_desc.get("attr"),
            )

            items.append(item)

        return items

    def _load_annotations(self, item: dict):
        parsed = item["annotations"]
        loaded = []

        for ann in parsed:
            ann_id = ann.get("id")
            ann_type = AnnotationType[ann["type"]]
            attributes = ann.get("attributes")
            group = ann.get("group")

            label_id = ann.get("label_id")
            z_order = ann.get("z_order")
            points = ann.get("points")

            if ann_type == AnnotationType.label:
                loaded.append(Label(label=label_id, id=ann_id, attributes=attributes, group=group))

            elif ann_type == AnnotationType.mask:
                rle = ann["rle"]
                rle["counts"] = rle["counts"].encode("ascii")
                loaded.append(
                    RleMask(
                        rle=rle,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.polyline:
                loaded.append(
                    PolyLine(
                        points,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.polygon:
                loaded.append(
                    Polygon(
                        points,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.bbox:
                x, y, w, h = ann["bbox"]
                loaded.append(
                    Bbox(
                        x,
                        y,
                        w,
                        h,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.points:
                loaded.append(
                    Points(
                        points,
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            elif ann_type == AnnotationType.caption:
                caption = ann.get("caption")
                loaded.append(Caption(caption, id=ann_id, attributes=attributes, group=group))

            elif ann_type == AnnotationType.cuboid_3d:
                loaded.append(
                    Cuboid3d(
                        ann.get("position"),
                        ann.get("rotation"),
                        ann.get("scale"),
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                    )
                )

            elif ann_type == AnnotationType.skeleton:
                loaded.append(
                    Skeleton(
                        elements=self._load_skeleton_elements_annotations(ann, label_id, points),
                        label=label_id,
                        id=ann_id,
                        attributes=attributes,
                        group=group,
                        z_order=z_order,
                    )
                )

            else:
                raise NotImplementedError()

        return loaded

    def _load_skeleton_elements_annotations(
        self, ann: dict, label_id: int, points: List[Union[float, int]]
    ) -> List[Points]:
        if len(points) % 3 != 0:
            raise InvalidAnnotationError(
                f"Points have invalid value count {len(points)}, "
                "which is not divisible by 3. Expected (x, y, visibility) triplets."
            )
        points_attributes = ann.get("points_attributes")
        if len(points) != len(points_attributes) * 3:
            raise InvalidAnnotationError(
                f"Points and Points_attributes lengths ({len(points)}, {len(points_attributes)}) do not match, "
                "for each triplet (x, y, visibility) in points there should be one dict in points_attributes."
            )

        label_category = self._categories[AnnotationType.label]
        sub_labels = self._categories[AnnotationType.points].items[label_id].labels
        return [
            Points(
                points=[x, y],
                visibility=[v],
                label=label_category.find(
                    name=sub_label, parent=label_category.items[label_id].name
                )[0],
                attributes=attrs,
            )
            for (x, y, v), sub_label, attrs in zip(
                take_by(points, 3), sub_labels, points_attributes
            )
        ]


class DatumaroImporter(Importer):
    @classmethod
    def detect(cls, context: FormatDetectionContext) -> None:
        annot_file = context.require_file("annotations/*.json")

        with context.probe_text_file(
            annot_file,
            'must be a JSON object with "categories" ' 'and "items" keys',
        ) as f:
            contents = parse_json(f.read())
            if not {"categories", "items"} <= contents.keys():
                raise Exception

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".json", "datumaro", dirname="annotations")
