# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Sequence, TypeVar

from datumaro.components.contexts.importer import _ImportFail
from datumaro.components.dataset_base import (
    CategoriesInfo,
    DatasetBase,
    DatasetInfo,
    DatasetItem,
    SubsetBase,
)

T = TypeVar("T")


def check_identicalness(seq: Sequence[T], raise_error_on_empty: bool = True) -> Optional[T]:
    if len(seq) == 0 and raise_error_on_empty:
        raise _ImportFail("It should not be empty.")
    elif len(seq) == 0 and not raise_error_on_empty:
        return None

    if seq.count(seq[0]) != len(seq):
        raise _ImportFail("All items in the sequence should be identical.")

    return seq[0]


class _ExtractorConcatenator(SubsetBase):
    """A simple class to merge not-intersecting single-subset extractors with the same subset"""

    def __init__(
        self,
        sources: Sequence[SubsetBase],
    ):
        if len(sources) == 0:
            raise _ImportFail("It should not be empty.")

        self._infos = check_identicalness([s.infos() for s in sources])
        self._categories = check_identicalness([s.categories() for s in sources])
        self._media_type = check_identicalness([s.media_type() for s in sources])
        self._subset = check_identicalness([s.subset for s in sources])

        ann_types = set()
        for source in sources:
            ann_types = ann_types.union(source.ann_types())
        self._ann_types = ann_types

        self._is_stream = check_identicalness([s.is_stream for s in sources])

        self._sources = sources

    def infos(self) -> DatasetInfo:
        return self._infos

    def categories(self) -> CategoriesInfo:
        return self._categories

    def __iter__(self) -> Iterator[DatasetItem]:
        for source in self._sources:
            yield from source

    def __len__(self) -> int:
        return sum(len(source) for source in self._sources)

    def get(self, id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        for source in self._sources:
            if item := source.get(id=id, subset=source.subset):
                return item

        return None

    @property
    def is_stream(self) -> bool:
        return self._is_stream


class ExtractorMerger(DatasetBase):
    """A simple class to merge not-intersecting single-subset extractors."""

    def __init__(
        self,
        sources: Sequence[SubsetBase],
    ):
        if len(sources) == 0:
            raise _ImportFail("It should not be empty.")

        self._infos = check_identicalness([s.infos() for s in sources])
        self._categories = check_identicalness([s.categories() for s in sources])
        self._media_type = check_identicalness([s.media_type() for s in sources])

        ann_types = set()
        for source in sources:
            ann_types = ann_types.union(source.ann_types())
        self._ann_types = ann_types

        self._is_stream = check_identicalness([s.is_stream for s in sources])

        subsets: Dict[str, List[SubsetBase]] = defaultdict(list)
        for source in sources:
            subsets[source.subset] += [source]

        self._subsets = {
            subset_name: _ExtractorConcatenator(sources) for subset_name, sources in subsets.items()
        }

    def infos(self) -> DatasetInfo:
        return self._infos

    def categories(self) -> CategoriesInfo:
        return self._categories

    def __iter__(self) -> Iterator[DatasetItem]:
        for subset in self._subsets.values():
            yield from subset

    def get_subset(self, name: str):
        if name not in self._subsets:
            raise KeyError(
                "Unknown subset '%s', available subsets: %s" % (name, set(self._subsets))
            )
        return self._subsets[name]

    def __len__(self) -> int:
        return sum(len(subset) for subset in self._subsets.values())

    def get(self, id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        if subset is not None and (source := self._subsets.get(subset)):
            if item := source.get(id, subset):
                return item

        for source in self._subsets.values():
            if item := source.get(id=id, subset=source.subset):
                return item

        return None

    @property
    def is_stream(self) -> bool:
        return self._is_stream
