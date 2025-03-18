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
        super().__init__(length=3, subsets=["train", "test", "foo"])
        self.iter_subset_call_count = 0
        self.iter_call_count = 0

    def __iter__(self):
        self.iter_call_count += 1
        yield from super().__iter__()

    def get_subset(self, name: str) -> IDataset:
        assert name in ["train", "test", "foo"]

        class _SubsetExtractor(SubsetBase):
            def __init__(self, parent):
                super().__init__(subset=name)
                self.parent = parent

            def __iter__(self):
                self.parent.iter_subset_call_count += 1
                yield DatasetItem(
                    id=f"{name}_1",
                    subset=name,
                    media=Image.from_numpy(data=np.ones((4, 2, 3))),
                    annotations=[],
                )

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
        # not all exporters can export in stream manner - skipping them
        assert "cannot export a dataset in a stream manner" in str(e)
        return

    # there was no full iterations
    assert extractor.iter_call_count == 0
    # each subset was iterated once
    assert extractor.iter_subset_call_count == len(extractor.subsets())
