import sys

import numpy as np
import pytest

from datumaro import AnnotationType, CategoriesInfo, LabelCategories
from datumaro.components.dataset import StreamDataset
from datumaro.components.dataset_base import DatasetItem, IDataset, StreamingDatasetBase, SubsetBase
from datumaro.components.environment import DEFAULT_ENVIRONMENT
from datumaro.components.errors import DatasetExportError
from datumaro.components.media import Image


class DummyStreamingExtractor(StreamingDatasetBase):
    def __init__(self):
        super().__init__(length=6, subsets=["train", "test", "foo"])
        self.iter_subset_call_dict = {subset: 0 for subset in self._subsets}
        self.iter_call_count = 0

    def __iter__(self):
        self.iter_call_count += 1
        yield from super().__iter__()

    def get_subset(self, name: str) -> IDataset:
        assert name in self._subsets

        class _SubsetExtractor(SubsetBase):
            def __init__(self, parent):
                super().__init__(subset=name)
                self.parent = parent

            def __iter__(self):
                self.parent.iter_subset_call_dict[name] += 1
                item = DatasetItem(
                    id=f"{name}_1",
                    subset=name,
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
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
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[],
                )
                assert sys.getrefcount(item) == 2

            @property
            def is_stream(self):
                return True

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
