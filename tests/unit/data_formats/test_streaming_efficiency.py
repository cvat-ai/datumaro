import os.path
import sys
from typing import Generator, Tuple
from unittest.mock import patch

import numpy as np
import pytest

from datumaro import AnnotationType, CategoriesInfo, LabelCategories
from datumaro.components import media
from datumaro.components.dataset import Dataset, StreamDataset
from datumaro.components.dataset_base import DatasetItem, StreamingDatasetBase, StreamingSubsetBase
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.errors import DatasetExportError


class DummyStreamingExtractor(StreamingDatasetBase):
    def __init__(self):
        super().__init__(length=6, subsets=["train", "test", "foo"])
        self.iter_subset_call_dict = {subset: 0 for subset in self._subsets}
        self.iter_call_count = 0

    def __iter__(self):
        self.iter_call_count += 1
        yield from super().__iter__()

    def get_subset(self, name: str) -> StreamingSubsetBase:
        assert name in self._subsets

        class _SubsetExtractor(StreamingSubsetBase):
            def __init__(self, parent):
                super().__init__(subset=name)
                self.parent = parent

            def __iter__(self):
                self.parent.iter_subset_call_dict[name] += 1
                item = DatasetItem(
                    id=f"{name}_1",
                    subset=name,
                    media=media.Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[],
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
                    annotations=[],
                )
                assert sys.getrefcount(item) == 2

            def ids(self) -> Generator[Tuple[str, str], None, None]:
                yield f"{name}_1", name
                yield f"{name}_2", name

        return _SubsetExtractor(self)

    def categories(self) -> CategoriesInfo:
        return {AnnotationType.label: LabelCategories.from_iterable(["a", "b", "c"])}


@pytest.mark.parametrize("exporter_cls", DEFAULT_ENVIRONMENT.exporters.items.values())
def test_streaming_exporters_only_iterate_items_once(test_dir, exporter_cls):
    extractor = DummyStreamingExtractor()
    dataset = StreamDataset(source=extractor)
    try:
        exporter_cls.convert(dataset, test_dir, stream=True)
    except DatasetExportError as e:
        assert "cannot export a dataset in a stream manner" in str(e)
        pytest.skip(f"{exporter_cls} does not support streaming")

    # there was no full iterations
    assert extractor.iter_call_count == 0
    # each subset was iterated once
    assert not {
        subset: call_count
        for subset, call_count in extractor.iter_subset_call_dict.items()
        if call_count != 1
    }


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
        # no inits on calling ids
        assert set(parsed_dataset.ids()) == set(fxt_dataset.ids())
        assert init_counter.count == len(fxt_dataset)

    # checking streaming importer
    with MediaElementInitCounter() as init_counter:
        parsed_dataset = StreamDataset.import_from(dataset_folder, format=import_format)
        # nothing initialized yet
        assert init_counter.count == 0
        # no inits on calling ids()
        assert set(parsed_dataset.ids()) == set(fxt_dataset.ids())
        assert init_counter.count == 0
        # inits on iteration
        assert len(list(parsed_dataset)) == len(fxt_dataset)
        assert init_counter.count == len(fxt_dataset)
        # inits again on iteration, i.e. not caching items
        assert len(list(parsed_dataset)) == len(fxt_dataset)
        assert init_counter.count == len(fxt_dataset) * 2
