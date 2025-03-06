import os.path as osp
from unittest import TestCase
from unittest.mock import Mock

import numpy as np

from datumaro.components.media import Image, ImageFromBytes
from datumaro.util.image import (
    encode_image,
    lazy_image,
    load_image_meta_file,
    save_image,
    save_image_meta_file,
)
from datumaro.util.image_cache import ImageCache

from tests.requirements import Requirements, mark_requirement
from tests.utils.test_utils import TestDir


class ImageCacheTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cache_works(self):
        with TestDir() as test_dir:
            image = np.ones((100, 100, 3), dtype=np.uint8)
            image_path = osp.join(test_dir, "image.jpg")
            save_image(image_path, image)

            caching_loader = lazy_image(image_path, cache=True)
            self.assertTrue(caching_loader() is caching_loader())

            non_caching_loader = lazy_image(image_path, cache=False)
            self.assertFalse(non_caching_loader() is non_caching_loader())

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_cache_fifo_displacement(self):
        capacity = 2
        cache = ImageCache(capacity)

        loaders = [
            lazy_image(None, loader=lambda p: object(), cache=cache) for _ in range(capacity + 1)
        ]

        first_request = [loader() for loader in loaders[1:]]
        loaders[0]()  # pop something from the cache

        second_request = [loader() for loader in loaders[2:]]
        second_request.insert(0, loaders[1]())

        matches = sum([a is b for a, b in zip(first_request, second_request)])
        self.assertEqual(matches, len(first_request) - 1)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_global_cache_is_accessible(self):
        loader = lazy_image(None, loader=lambda p: object())

        self.assertTrue(loader() is loader())
        self.assertEqual(ImageCache.get_instance().size(), 1)

    def setUp(self) -> None:
        ImageCache.get_instance().clear()
        return super().setUp()

    def tearDown(self) -> None:
        ImageCache.get_instance().clear()
        return super().tearDown()


class ImageTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_can_report_cached_size(self):
        data = np.ones((5, 6, 3))

        image = Image.from_numpy(data=lambda _: data, size=(2, 4))

        self.assertEqual((2, 4), image.size)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ctors(self):
        with TestDir() as test_dir:
            path = osp.join(test_dir, "path.png")
            image = np.ones([2, 4, 3])
            save_image(path, image)

            for args in [
                {"data": image},
                {"data": image, "ext": "png"},
                {"data": image, "ext": "png", "size": (2, 4)},
                {"data": lambda: image},
                {"data": lambda: image, "ext": "jpg"},
                {"path": path},
                {"path": path, "size": (2, 4)},
            ]:
                with self.subTest(**args):
                    assert "path" not in args or "data" not in args
                    if "path" in args:
                        img = Image.from_file(**args)
                    else:
                        img = Image.from_numpy(**args)

                    self.assertTrue(img.has_data)
                    np.testing.assert_array_equal(img.data, image)
                    self.assertEqual(img.size, tuple(image.shape[:2]))

            with self.subTest():
                img = Image.from_file(path="somepath", size=(2, 4))
                self.assertEqual(img.size, (2, 4))

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ctor_errors(self):
        with self.subTest("no data specified"):
            with self.assertRaisesRegex(Exception, "Directly initalizing"):
                Image(ext="jpg")


class BytesImageTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_lazy_image_shape(self):
        data = encode_image(np.ones((5, 6, 3)), "png")

        image_lazy = ImageFromBytes(data=data, size=(2, 4))
        image_eager = ImageFromBytes(data=data)

        self.assertEqual((2, 4), image_lazy.size)
        self.assertEqual((5, 6), image_eager.size)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ctors(self):
        with TestDir() as test_dir:
            path = osp.join(test_dir, "path.png")
            image = np.ones([2, 4, 3])
            image_bytes = encode_image(image, "png")

            for args in [
                {"data": image_bytes},
                {"data": lambda: image_bytes},
                {"data": lambda: image_bytes, "ext": ".jpg"},
            ]:
                with self.subTest(**args):
                    img = ImageFromBytes(**args)
                    # pylint: disable=pointless-statement
                    self.assertEqual("data" in args, img.has_data)
                    if img.has_data:
                        np.testing.assert_array_equal(img.data, image)
                        self.assertEqual(img.bytes, image_bytes)
                    if "size" in args:
                        self.assertEqual(img.size, args["size"])
                    if "ext" in args or "path" in args:
                        self.assertEqual(img.ext, args.get("ext", ".png"))
                    # pylint: enable=pointless-statement

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ext_detection(self):
        image_data = np.zeros((3, 4))

        for ext in (".bmp", ".jpg", ".png"):
            with self.subTest(ext=ext):
                image = ImageFromBytes(data=encode_image(image_data, ext))
                self.assertEqual(image.ext, ext)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_ext_detection_failure(self):
        image_bytes = b"\xff" * 10  # invalid image
        image = ImageFromBytes(data=image_bytes)
        self.assertEqual(image.ext, None)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_no_excess_decode_on_image_save(self):
        with TestDir() as test_dir:
            image_np = np.ones([2, 4, 3])

            extensions = ["png", "bmp", "jpg"]
            for source_ext, save_ext in zip(extensions, extensions):
                with self.subTest(source_ext=source_ext, save_ext=save_ext):
                    image_bytes = encode_image(image_np, source_ext)
                    img = Image.from_bytes(data=image_bytes)
                    mimg = Mock(wraps=img)
                    mimg.save(osp.join(test_dir, f"path_{source_ext}.{save_ext}"))
                    assert mimg.data.call_count == 0 if source_ext == save_ext else 1


class ImageMetaTest(TestCase):
    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_loading(self):
        meta_file_contents = r"""
        # this is a comment

        a 123 456
        'b c' 10 20 # inline comment
        """

        meta_expected = {
            "a": (123, 456),
            "b c": (10, 20),
        }

        with TestDir() as test_dir:
            meta_path = osp.join(test_dir, "images.meta")

            with open(meta_path, "w") as meta_file:
                meta_file.write(meta_file_contents)

            meta_loaded = load_image_meta_file(meta_path)

        self.assertEqual(meta_loaded, meta_expected)

    @mark_requirement(Requirements.DATUM_GENERAL_REQ)
    def test_saving(self):
        meta_original = {
            "a": (123, 456),
            "b c": (10, 20),
        }

        with TestDir() as test_dir:
            meta_path = osp.join(test_dir, "images.meta")

            save_image_meta_file(meta_original, meta_path)
            meta_reloaded = load_image_meta_file(meta_path)

        self.assertEqual(meta_reloaded, meta_original)
