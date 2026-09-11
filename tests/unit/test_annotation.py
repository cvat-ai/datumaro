# Copyright (C) 2026 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import numpy as np
import pytest

from datumaro.components.annotation import ExtractedMask, Mask, RleMask
from datumaro.util import mask_tools

from tests.requirements import Requirements, mark_bug

_IMAGE = np.array([[1]], dtype=np.uint8)


def _rle_mask(**kwargs) -> RleMask:
    return RleMask(
        rle=mask_tools.to_uncompressed_rle(mask_tools.mask_to_rle(_IMAGE), width=1, height=1),
        **kwargs,
    )


class AnnotationEqualityTest:
    @mark_bug(Requirements.DATUM_CVAT_AI_BUG_133)
    def test_mask_equals_rle_mask_when_shared_fields_match(self):
        """
        <b>Description:</b>
        Mask and RleMask with the same image and parent fields compare equal.

        <b>Expected results:</b>
        Equality is True in both directions.

        <b>Steps:</b>
        1. Create a Mask and an RleMask with matching image, id, and related fields
        2. Compare them in both orders
        """
        kwargs = dict(id=1, label=3, z_order=2, group=4, object_id=5, attributes={"a": 1})
        mask = Mask(_IMAGE, **kwargs)
        rle_mask = _rle_mask(**kwargs)

        assert mask == rle_mask
        assert rle_mask == mask

    @mark_bug(Requirements.DATUM_CVAT_AI_BUG_133)
    @pytest.mark.parametrize(
        "field, value_a, value_b",
        [
            ("id", 1, 2),
            ("group", 1, 2),
            ("object_id", 1, 2),
            ("attributes", {"occluded": True}, {"occluded": False}),
            ("label", 0, 1),
            ("z_order", 0, 1),
        ],
    )
    def test_mask_and_rle_mask_are_unequal_when_fields_differ(self, field, value_a, value_b):
        """
        <b>Description:</b>
        Mask vs RleMask equality includes Annotation and Mask fields, not only the image.

        <b>Expected results:</b>
        Annotations that differ only in a parent or Mask field are unequal.

        <b>Steps:</b>
        1. Create a Mask and an RleMask with the same image
        2. Differ a single shared field
        3. Compare them in both orders
        """
        mask = Mask(_IMAGE, **{field: value_a})
        rle_mask = _rle_mask(**{field: value_b})

        assert mask != rle_mask
        assert rle_mask != mask

    @mark_bug(Requirements.DATUM_CVAT_AI_BUG_133)
    @pytest.mark.parametrize(
        "field, value_a, value_b",
        [
            ("id", 1, 2),
            ("group", 1, 2),
            ("object_id", 1, 2),
            ("attributes", {"occluded": True}, {"occluded": False}),
            ("label", 0, 1),
            ("z_order", 0, 1),
        ],
    )
    def test_rle_masks_are_unequal_when_fields_differ(self, field, value_a, value_b):
        """
        <b>Description:</b>
        Two RleMask annotations compare parent and Mask fields, not only the RLE payload.

        <b>Expected results:</b>
        RleMasks that differ only in a parent or Mask field are unequal.

        <b>Steps:</b>
        1. Create two RleMasks with the same RLE payload
        2. Differ a single shared field
        3. Compare them
        """
        assert _rle_mask(**{field: value_a}) != _rle_mask(**{field: value_b})

    @mark_bug(Requirements.DATUM_CVAT_AI_BUG_133)
    def test_mask_and_extracted_mask_are_unequal_when_ids_differ(self):
        """
        <b>Description:</b>
        Mask vs ExtractedMask equality includes Annotation fields.

        <b>Expected results:</b>
        A Mask and ExtractedMask with the same image but different ids are unequal.

        <b>Steps:</b>
        1. Create a Mask and an ExtractedMask that decode to the same binary image
        2. Give them different ids
        3. Compare them in both orders
        """
        index_mask = np.array([[1]], dtype=np.uint8)
        mask = Mask(index_mask == 1, id=1)
        extracted = ExtractedMask(index_mask=index_mask, index=1, id=2)

        assert mask != extracted
        assert extracted != mask
