# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import math
import os.path as osp
from glob import iglob

import numpy as np

from datumaro.components.annotation import AnnotationType, Cuboid3d, LabelCategories
from datumaro.components.dataset_base import DatasetItem, SubsetBase
from datumaro.components.importer import Importer
from datumaro.components.media import CameraCalibration, Image, PointCloud
from datumaro.util import parse_json_file
from datumaro.util.image import find_images
from datumaro.util.os_util import safe_join

from .format import PointCloudPath


def _parse_calibration_matrix(
    matrix, *, name: str, expected_length: int, sidecar_path: str
) -> tuple[float, ...]:
    if not isinstance(matrix, list) or len(matrix) != expected_length:
        raise ValueError(
            f"Invalid calibration in {sidecar_path}: "
            f"'{name}' must contain {expected_length} values"
        )

    values = []
    for value in matrix:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Invalid calibration in {sidecar_path}: "
                f"'{name}' must contain only finite numbers"
            )

        try:
            value = float(value)
        except OverflowError:
            raise ValueError(
                f"Invalid calibration in {sidecar_path}: "
                f"'{name}' must contain only finite numbers"
            )

        if not math.isfinite(value):
            raise ValueError(
                f"Invalid calibration in {sidecar_path}: "
                f"'{name}' must contain only finite numbers"
            )
        values.append(value)

    return tuple(values)


def _load_camera_calibration(image_path: str) -> CameraCalibration | None:
    sidecar_path = image_path + ".json"
    if not osp.isfile(sidecar_path):
        return None

    sidecar = parse_json_file(sidecar_path)
    if not isinstance(sidecar, dict):
        raise ValueError(f"Invalid calibration in {sidecar_path}: the root value must be an object")

    meta = sidecar.get("meta")
    if meta is None:
        return None
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid calibration in {sidecar_path}: 'meta' must be an object")

    sensors_data = meta.get("sensorsData")
    if sensors_data is None:
        return None
    if not isinstance(sensors_data, dict):
        raise ValueError(
            f"Invalid calibration in {sidecar_path}: 'meta.sensorsData' must be an object"
        )

    intrinsic_matrix = _parse_calibration_matrix(
        sensors_data.get("intrinsicMatrix"),
        name="meta.sensorsData.intrinsicMatrix",
        expected_length=9,
        sidecar_path=sidecar_path,
    )
    extrinsic_matrix = _parse_calibration_matrix(
        sensors_data.get("extrinsicMatrix"),
        name="meta.sensorsData.extrinsicMatrix",
        expected_length=12,
        sidecar_path=sidecar_path,
    )

    try:
        np.linalg.inv(np.asarray(intrinsic_matrix).reshape(3, 3))
    except np.linalg.LinAlgError:
        raise ValueError(
            f"Invalid calibration in {sidecar_path}: "
            "'meta.sensorsData.intrinsicMatrix' must be invertible"
        )

    rotation = np.asarray(extrinsic_matrix).reshape(3, 4)[:, :3]
    # check orthogonality
    if not np.allclose(
        rotation.T @ rotation, np.identity(3), atol=1e-5, rtol=0
    ) or not np.isclose(  # exclude mirror/reflection matrices
        np.linalg.det(rotation), 1, atol=1e-5, rtol=0
    ):
        raise ValueError(
            f"Invalid calibration in {sidecar_path}: "
            "'meta.sensorsData.extrinsicMatrix' must contain a rigid rotation and translation"
        )

    return CameraCalibration(
        intrinsic_matrix=intrinsic_matrix,
        extrinsic_matrix=extrinsic_matrix,
    )


class SuperviselyPointCloudBase(SubsetBase):
    NAME = "sly_pointcloud"
    _SUPPORTED_SHAPES = "cuboid"

    def __init__(self, path, subset=None):
        if not osp.isfile(path):
            raise FileNotFoundError("Expected a path to 'meta.json', " "got '%s'" % path)

        rootdir = osp.abspath(osp.dirname(path))
        self._rootdir = rootdir

        super().__init__(subset=subset, media_type=PointCloud)

        items, categories = self._parse(rootdir)
        self._items = list(self._load_items(items).values())
        self._categories = categories

    @classmethod
    def _parse(cls, rootpath):
        mapping = parse_json_file(osp.join(rootpath, PointCloudPath.KEY_ID_FILE))
        meta = parse_json_file(osp.join(rootpath, PointCloudPath.META_FILE))

        label_cat = LabelCategories()
        for label in meta.get("classes", []):
            label_cat.add(label["title"])

        tags = {}
        for tag in meta.get("tags", []):
            # See reference at:
            # https://github.com/supervisely/supervisely/blob/047e52ebe407cfee61464c1bd0beb9c906892253/supervisely_lib/annotation/tag_meta.py#L139
            tags[tag["name"]] = tag

            applicable_to = tag.get("applicable_type", "all")
            if applicable_to == "imagesOnly":
                continue  # an image attribute
            elif applicable_to not in {"all", "objectsOnly"}:
                raise Exception("Unexpected tag 'applicable_type' value '%s'" % applicable_to)

            applicable_classes = tag.get("classes", [])
            if not applicable_classes:
                label_cat.attributes.add(tag["name"])
            else:
                for label_name in applicable_classes:
                    _, label = label_cat.find(label_name)
                    if label is None:
                        raise Exception("Unknown class for tag '%s'" % label_name)

                    label.attributes.add(tag["name"])

        categories = {AnnotationType.label: label_cat}

        def _get_label_attrs(label_id):
            attrs = set(label_cat.attributes)
            attrs.update(label_cat[label_id].attributes)
            return attrs

        def _parse_tag(tag):
            if tag["value"] == "true":
                value = True
            elif tag["value"] == "false":
                value = False
            else:
                value = tag["value"]
            return value

        ann_dir = osp.join(rootpath, PointCloudPath.BASE_DIR, PointCloudPath.ANNNOTATION_DIR)
        items = {}
        for ann_file in iglob(osp.join(ann_dir, "**", "*.json"), recursive=True):
            ann_data = parse_json_file(ann_file)

            objects = {}
            for obj in ann_data["objects"]:
                obj["id"] = mapping["objects"][obj["key"]]
                objects[obj["key"]] = obj

            frame_attributes = {"description": ann_data.get("description", "")}
            for tag in ann_data["tags"]:
                frame_attributes[tag["name"]] = _parse_tag(tag)

            frame = mapping["videos"][ann_data["key"]]
            frame_desc = items.setdefault(
                frame,
                {
                    "name": osp.splitext(osp.relpath(ann_file, ann_dir))[0],
                    "annotations": [],
                    "attributes": frame_attributes,
                },
            )

            for figure in ann_data["figures"]:
                geometry = {
                    dst_field: [
                        float(figure["geometry"][src_field][axis]) for axis in ["x", "y", "z"]
                    ]
                    for src_field, dst_field in {
                        "position": "position",
                        "rotation": "rotation",
                        "dimensions": "scale",
                    }.items()
                }

                ann_id = mapping["figures"][figure["key"]]

                obj = objects[figure["objectKey"]]
                label = categories[AnnotationType.label].find(obj["classTitle"])[0]

                attributes = {}
                attributes["track_id"] = obj["id"]
                for tag in obj.get("tags", []):
                    attributes[tag["name"]] = _parse_tag(tag)
                for attr in _get_label_attrs(label):
                    if attr in attributes:
                        continue
                    if tags[attr]["value_type"] == "any_string":
                        value = ""
                    elif tags[attr]["value_type"] == "oneof_string":
                        value = (tags[attr]["values"] or [""])[0]
                    elif tags[attr]["value_type"] == "any_number":
                        value = 0
                    else:
                        value = None
                    attributes[attr] = value

                shape = Cuboid3d(**geometry, label=label, id=ann_id, attributes=attributes)

                frame_desc["annotations"].append(shape)

        return items, categories

    def _load_items(self, parsed):
        for frame_id, frame_desc in parsed.items():
            pcd_name = frame_desc["name"]
            name = osp.splitext(pcd_name)[0]
            pcd_path = safe_join(
                self._rootdir, PointCloudPath.BASE_DIR, PointCloudPath.POINT_CLOUD_DIR, pcd_name
            )
            assert pcd_path.endswith(".pcd"), pcd_path

            related_images_dir = safe_join(
                self._rootdir,
                PointCloudPath.BASE_DIR,
                PointCloudPath.RELATED_IMAGES_DIR,
                name + "_pcd",
            )
            related_images = None
            if osp.isdir(related_images_dir):
                related_images = []
                for image_path in find_images(related_images_dir):
                    camera_calibration = _load_camera_calibration(image_path)
                    sidecar_path = f"{image_path}.json"
                    related_images.append(
                        Image.from_file(
                            path=image_path,
                            camera_calibration=camera_calibration,
                            auxiliary_files=[sidecar_path] if osp.isfile(sidecar_path) else [],
                        )
                    )

            parsed[frame_id] = DatasetItem(
                id=name,
                subset=self._subset,
                media=PointCloud.from_file(pcd_path, extra_images=related_images),
                annotations=frame_desc.get("annotations"),
                attributes={"frame": int(frame_id), **frame_desc["attributes"]},
            )

        return parsed


class SuperviselyPointCloudImporter(Importer):
    NAME = "sly_pointcloud"

    @classmethod
    def find_sources(cls, path):
        return cls._find_sources_recursive(path, ".json", "sly_pointcloud", filename="meta")
