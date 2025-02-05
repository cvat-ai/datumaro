import os.path as osp
from itertools import product
from unittest import TestCase

import numpy as np

import datumaro.util.image as image_module

from tests.requirements import Requirements, mark_requirement
from tests.utils.test_utils import TestDir


class ImageOperationsTest(TestCase):
    def setUp(self):
        self.default_backend = image_module.IMAGE_BACKEND.get()

    def tearDown(self):
        image_module.IMAGE_BACKEND.set(self.default_backend)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_save_and_load_backends(self):
        backends = image_module.ImageBackend
        for save_backend, load_backend, c in product(backends, backends, [1, 3]):
            with TestDir() as test_dir:
                if c == 1:
                    src_image = np.random.randint(0, 255 + 1, (2, 4))
                else:
                    src_image = np.random.randint(0, 255 + 1, (2, 4, c))
                path = osp.join(test_dir, "img.png")  # lossless

                image_module.IMAGE_BACKEND.set(save_backend)
                image_module.save_image(path, src_image, jpeg_quality=100)

                image_module.IMAGE_BACKEND.set(load_backend)
                dst_image = image_module.load_image(path)

                # If image_module.IMAGE_COLOR_CHANNEL.get() == image_module.ImageColorChannel.UNCHANGED
                # OpenCV will read an image as BGR(A), but PIL will read an image as RGB(A).
                if (
                    c == 3
                    and load_backend == image_module.ImageBackend.PIL
                    and image_module.IMAGE_COLOR_CHANNEL.get()
                    == image_module.ImageColorChannel.UNCHANGED
                ):
                    dst_image = np.flip(dst_image, -1)

                self.assertTrue(
                    np.array_equal(src_image, dst_image),
                    "save: %s, load: %s" % (save_backend, load_backend),
                )

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_encode_and_decode_backends(self):
        backends = image_module.ImageBackend
        for save_backend, load_backend, c in product(backends, backends, [1, 3]):
            if c == 1:
                src_image = np.random.randint(0, 255 + 1, (2, 4))
            else:
                src_image = np.random.randint(0, 255 + 1, (2, 4, c))

            image_module.IMAGE_BACKEND.set(save_backend)
            buffer = image_module.encode_image(src_image, ".png", jpeg_quality=100)  # lossless

            image_module.IMAGE_BACKEND.set(load_backend)
            dst_image = image_module.decode_image(buffer)

            # If image_module.IMAGE_COLOR_CHANNEL.get() == image_module.ImageColorChannel.UNCHANGED
            # OpenCV will read an image as BGR(A), but PIL will read an image as RGB(A).
            if (
                c == 3
                and load_backend == image_module.ImageBackend.PIL
                and image_module.IMAGE_COLOR_CHANNEL.get()
                == image_module.ImageColorChannel.UNCHANGED
            ):
                dst_image = np.flip(dst_image, -1)

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
