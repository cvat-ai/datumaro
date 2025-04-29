import os.path
import sys
from unittest.mock import patch

import numpy as np
import pytest

from datumaro import AnnotationType, CategoriesInfo, LabelCategories
from datumaro.components import media
from datumaro.components.annotation import Annotations
from datumaro.components.dataset import Dataset, StreamDataset
from datumaro.components.dataset_base import DatasetBase, DatasetItem
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.errors import DatasetExportError
from datumaro.plugins.data_formats.coco.exporter import CocoImageInfoExporter


class DummyStreamingExtractor(DatasetBase):
    def __init__(self):
        super().__init__(subsets=["train", "test", "foo"])
        self.ann_init_counter = 0

    def _generate_anns(self):
        self.ann_init_counter += 1
        return []

    def __iter__(self):
        for name in self._subsets:
            item = DatasetItem(
                id=f"{name}_1",
                subset=name,
                media=media.Image.from_numpy(data=np.ones((4, 2, 3))),
                annotations=self._generate_anns,
            )
            # counting references to make sure that exporter is actually streaming

            # before yielded, references are only here
            assert sys.getrefcount(item) == 2

            # after yielded, there are more references (e.g. where it's yielded from)
            # number of references doesn't have to increase in general,
            # but it should due to how our code works
            yield item
            assert sys.getrefcount(item) > 2

            # after next item yielded, ref count is 2 again - i.e. item was not saved anywhere
            yield DatasetItem(
                id=f"{name}_2",
                subset=name,
                media=media.Image.from_numpy(data=np.ones((4, 2, 3))),
                annotations=self._generate_anns,
            )
            assert sys.getrefcount(item) == 2

    @property
    def is_stream(self) -> bool:
        return True

    def categories(self) -> CategoriesInfo:
        return {AnnotationType.label: LabelCategories.from_iterable(["a", "b", "c"])}


@pytest.mark.parametrize("exporter_cls", DEFAULT_ENVIRONMENT.exporters.items.values())
def test_streaming_exporters_only_init_annotations_once(test_dir, exporter_cls):
    extractor = DummyStreamingExtractor()
    dataset = StreamDataset(source=extractor)
    try:
        exporter_cls.convert(dataset, test_dir, stream=True)
    except DatasetExportError as e:
        assert "cannot export a dataset in a stream manner" in str(e)
        pytest.skip(f"{exporter_cls} does not support streaming")

    if exporter_cls is CocoImageInfoExporter:
        assert extractor.ann_init_counter == 0
        return

    # annotations initialization was done only once per item
    assert extractor.ann_init_counter == len(extractor)


@pytest.fixture(scope="session")
def fxt_dataset():
    subsets = ["train", "test", "val"]
    return Dataset.from_iterable(
        [
            DatasetItem(
                id=f"item_{index}",
                subset=subsets[index % len(subsets)],
                media=media.Image.from_numpy(data=np.ones((4, 2, 3))),
                annotations=[],
            )
            for index in range(10)
        ],
        categories=["aaa", "bbbb"],
    )


class MediaElementInitCounter:
    def __init__(self):
        self.count = 0

    def __enter__(self):
        orig_init = media.MediaElement.__init__

        def mock_init_func(instance):
            self.count += 1
            orig_init(instance)

        self._patch = patch.object(media.MediaElement, "__init__", mock_init_func)
        self._patch.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._patch.stop()


@pytest.mark.parametrize("export_format", DEFAULT_ENVIRONMENT.exporters.items.keys())
def test_streaming_importers(test_dir, export_format, fxt_dataset):
    import_format = {
        "mots_png": "mots",
        "mot_seq_gt": "mot_seq",
    }.get(export_format, export_format)
    assert import_format in DEFAULT_ENVIRONMENT.importers, sorted(
        DEFAULT_ENVIRONMENT.importers.items.keys()
    )

    if not DEFAULT_ENVIRONMENT.make_importer(import_format).can_stream:
        pytest.skip(f"Importer for '{import_format}' can not stream")

    dataset_folder = os.path.join(test_dir, "dataset")
    fxt_dataset.export(dataset_folder, format=export_format, save_media=True)

    # checking baseline non-streaming importer
    with MediaElementInitCounter() as init_counter:
        parsed_dataset = Dataset.import_from(dataset_folder, format=import_format)
        assert len(list(parsed_dataset)) == len(fxt_dataset)
        # after first iteration all items are initialized
        assert init_counter.count == len(fxt_dataset)
        # no inits on second iteration
        assert len(list(parsed_dataset)) == len(fxt_dataset)
        assert init_counter.count == len(fxt_dataset)

    # checking streaming importer
    with MediaElementInitCounter() as init_counter:
        parsed_dataset = StreamDataset.import_from(dataset_folder, format=import_format)
        # nothing initialized yet
        assert init_counter.count == 0

        for item in parsed_dataset:
            # annotations are not parsed if we do not access them
            assert not item.annotations_are_initialized
            # annotations are parsed if we access them
            assert isinstance(item.annotations, Annotations)
            assert item.annotations_are_initialized
        assert init_counter.count == len(fxt_dataset)

        # inits again on iteration, i.e. not caching items
        assert len(list(parsed_dataset)) == len(fxt_dataset)
        assert init_counter.count == len(fxt_dataset) * 2

        # subsets can be accessed
        for subset in parsed_dataset.subsets().values():
            assert subset.is_stream
            for item in subset:
                # annotations are not parsed if we do not access them
                assert not item.annotations_are_initialized
                # annotations are parsed if we access them
                assert isinstance(item.annotations, Annotations)
                assert item.annotations_are_initialized
