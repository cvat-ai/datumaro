# Copyright (C) 2019-2024 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os.path as osp
from itertools import product
from unittest import TestCase

import numpy as np
import PIL
import pytest

import datumaro.util.image as image_module

from ..requirements import Requirements, mark_requirement

from tests.utils.test_utils import TestDir


def generate_test_img(channels: int) -> np.ndarray:
    size = (5, 4, channels) if channels > 1 else (5, 4)
    return np.random.randint(low=0, high=256, size=size, dtype=np.uint8)


class ImageOperationsTest(TestCase):
    def setUp(self):
        self.default_backend = image_module.IMAGE_BACKEND.get()

    def tearDown(self):
        image_module.IMAGE_BACKEND.set(self.default_backend)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_and_load_backends(self):
        backends = image_module.ImageBackend
        for save_backend, load_backend, c in product(backends, backends, [1, 3, 4]):
            with TestDir() as test_dir:
                src_image = generate_test_img(c)
                path = osp.join(test_dir, "img.png")  # lossless

                image_module.IMAGE_BACKEND.set(save_backend)
                image_module.save_image(path, src_image)

                image_module.IMAGE_BACKEND.set(load_backend)
                dst_image = image_module.load_image(path)

                # If image_module.IMAGE_COLOR_CHANNEL.get() == image_module.ImageColorChannel.UNCHANGED
                # OpenCV will read an image as BGR(A), but PIL will read an image as RGB(A).
                if c in [3, 4] and (
                    load_backend == image_module.ImageBackend.PIL
                    and image_module.IMAGE_COLOR_CHANNEL.get()
                    == image_module.ImageColorChannel.UNCHANGED
                    or image_module.IMAGE_COLOR_CHANNEL.get()
                    == image_module.ImageColorChannel.COLOR_RGB
                ):
                    dst_image[..., :3] = dst_image[..., 2::-1]  # to bgr

                self.assertTrue(
                    np.array_equal(src_image, dst_image),
                    "save: %s, load: %s" % (save_backend, load_backend),
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_encode_and_decode_backends(self):
        backends = image_module.ImageBackend
        for save_backend, load_backend, c in product(backends, backends, [1, 3, 4]):
            src_image = generate_test_img(c)

            image_module.IMAGE_BACKEND.set(save_backend)
            buffer = image_module.encode_image(src_image, ".png")  # lossless

            image_module.IMAGE_BACKEND.set(load_backend)
            dst_image = image_module.decode_image(buffer)

            # If image_module.IMAGE_COLOR_CHANNEL.get() == image_module.ImageColorChannel.UNCHANGED
            # OpenCV will read an image as BGR(A), but PIL will read an image as RGB(A).
            if c in [3, 4] and (
                load_backend == image_module.ImageBackend.PIL
                and image_module.IMAGE_COLOR_CHANNEL.get()
                == image_module.ImageColorChannel.UNCHANGED
                or image_module.IMAGE_COLOR_CHANNEL.get()
                == image_module.ImageColorChannel.COLOR_RGB
            ):
                dst_image[..., :3] = dst_image[..., 2::-1]  # to bgr

            self.assertTrue(
                np.array_equal(src_image, dst_image),
                "save: %s, load: %s" % (save_backend, load_backend),
            )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_image_to_inexistent_dir_raises_error(self):
        with self.assertRaises(FileNotFoundError):
            image_module.save_image("some/path.jpg", np.ones((5, 4, 3)), create_dir=False)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_image_can_create_dir(self):
        with TestDir() as test_dir:
            path = osp.join(test_dir, "some", "path.jpg")
            image_module.save_image(path, np.ones((5, 4, 3)), create_dir=True)
            self.assertTrue(osp.isfile(path))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_load_image_with_exif_info(self):
        with TestDir() as test_dir:
            image_path = osp.join(test_dir, "1.jpg")
            image_module.save_image(image_path, np.ones((15, 10, 3)))

            img = PIL.Image.open(image_path)
            exif = img.getexif()
            exif.update([(PIL.ExifTags.Base.Orientation, 6)])
            img.save(image_path, exif=exif)

            for load_backend in image_module.ImageBackend:
                image_module.IMAGE_BACKEND.set(load_backend)
                img = image_module.load_image(image_path)
                assert img.shape == (10, 15, 3)


class ImageDecodeTest:
    @pytest.mark.parametrize("image_backend", image_module.ImageBackend)
    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_decode_image_context(self, image_backend: image_module.ImageBackend, channels: int):
        original_image = generate_test_img(channels)
        img_bytes = image_module.encode_image(original_image, ".png")

        if channels == 1:
            expected_bgr_image = np.repeat(original_image[:, :, np.newaxis], 3, axis=2)
        else:
            expected_bgr_image = original_image[:, :, :3]

        # 3 channels from ImageColorScale.COLOR_BGR
        with image_module.decode_image_context(
            image_backend, image_module.ImageColorChannel.COLOR_BGR
        ):
            img_decoded = image_module.decode_image(img_bytes)
            assert img_decoded.shape[-1] == 3
            assert np.allclose(expected_bgr_image, img_decoded)

        # 3 channels from ImageColorScale.COLOR_RGB
        with image_module.decode_image_context(
            image_backend, image_module.ImageColorChannel.COLOR_RGB
        ):
            img_decoded = image_module.decode_image(img_bytes)
            assert img_decoded.shape[-1] == 3
            assert np.allclose(expected_bgr_image[:, :, ::-1], img_decoded)

        # 1 (without an extra dim), 3 or 4 channels from ImageColorScale.UNCHANGED
        with image_module.decode_image_context(
            image_backend, image_module.ImageColorChannel.UNCHANGED
        ):
            img_decoded = image_module.decode_image(img_bytes)
            assert img_decoded.shape == original_image.shape

            if image_backend == image_module.ImageBackend.PIL and channels != 1:
                # PIL returns RGB(A)
                img_decoded[:, :, :3] = img_decoded[:, :, 2::-1] # to bgr

            assert np.allclose(original_image, img_decoded)
