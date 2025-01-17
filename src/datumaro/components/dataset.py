# Copyright (C) 2020-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from __future__ import annotations

import inspect
import logging as log
import os
import os.path as osp
import warnings
from contextlib import contextmanager
from copy import copy
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Type, Union

from datumaro.components.annotation import AnnotationType, LabelCategories
from datumaro.components.config_model import Source
from datumaro.components.contexts.importer import ImportErrorPolicy, _ImportFail
from datumaro.components.dataset_base import (
    CategoriesInfo,
    DatasetBase,
    DatasetInfo,
    DatasetItem,
    IDataset,
    ImportContext,
)
from datumaro.components.dataset_storage import DatasetPatch, DatasetStorage
from datumaro.components.environment import Environment
from datumaro.components.errors import (
    MultipleFormatsMatchError,
    NoMatchingFormatsError,
    UnknownFormatError,
)
from datumaro.components.exporter import ExportContext, Exporter, ExportErrorPolicy, _ExportFail
from datumaro.components.filter import XPathAnnotationsFilter, XPathDatasetFilter
from datumaro.components.launcher import Launcher
from datumaro.components.media import Image, MediaElement
from datumaro.components.progress_reporting import NullProgressReporter, ProgressReporter
from datumaro.components.transformer import ItemTransform, ModelTransform, Transform
from datumaro.util.definitions import DEFAULT_SUBSET_NAME
from datumaro.util.log_utils import logging_disabled
from datumaro.util.os_util import rmtree
from datumaro.util.scope import on_error_do, scoped

DEFAULT_FORMAT = "datumaro"


class DatasetSubset(IDataset):  # non-owning view
    def __init__(self, parent: Dataset, name: str):
        super().__init__()
        self.parent = parent
        self.name = name

    def __iter__(self):
        yield from self.parent._data.get_subset(self.name)

    def __len__(self):
        return len(self.parent._data.get_subset(self.name))

    def put(self, item):
        return self.parent.put(item, subset=self.name)

    def get(self, id, subset=None):
        assert (subset or DEFAULT_SUBSET_NAME) == (self.name or DEFAULT_SUBSET_NAME)
        return self.parent.get(id, subset=self.name)

    def remove(self, id, subset=None):
        assert (subset or DEFAULT_SUBSET_NAME) == (self.name or DEFAULT_SUBSET_NAME)
        return self.parent.remove(id, subset=self.name)

    def get_subset(self, name):
        assert (name or DEFAULT_SUBSET_NAME) == (self.name or DEFAULT_SUBSET_NAME)
        return self

    def subsets(self):
        if (self.name or DEFAULT_SUBSET_NAME) == DEFAULT_SUBSET_NAME:
            return self.parent.subsets()
        return {self.name: self}

    def infos(self) -> DatasetInfo:
        return {}

    def categories(self):
        return self.parent.categories()

    def media_type(self):
        return self.parent.media_type()

    def ann_types(self) -> Set[AnnotationType]:
        return set()

    def as_dataset(self) -> Dataset:
        return Dataset.from_extractors(self, env=self.parent.env)


class Dataset(IDataset):
    """
    Represents a dataset, contains metainfo about labels and dataset items.
    Provides iteration and access options to dataset elements.

    By default, all operations are done lazily, it can be changed by
    modifying the `eager` property and by using the `eager_mode`
    context manager.

    Dataset is supposed to have a single media type for its items. If the
    dataset is filled manually or from extractors, and media type does not
    match, an error is raised.
    """

    _global_eager: bool = False

    @classmethod
    def from_iterable(
        cls,
        iterable: Iterable[DatasetItem],
        categories: Union[CategoriesInfo, List[str], None] = None,
        *,
        env: Optional[Environment] = None,
        media_type: Type[MediaElement] = Image,
    ) -> Dataset:
        """
        Creates a new dataset from an iterable object producing dataset items -
        a generator, a list etc. It is a convenient way to create and fill
        a custom dataset.

        Parameters:
            iterable: An iterable which returns dataset items
            categories: A simple list of labels or complete information
                about labels. If not specified, an empty list of labels
                is assumed.
            media_type: Media type for the dataset items. If the sequence
                contains items with mismatching media type, an error is
                raised during caching
            env: A context for plugins, which will be used for this dataset.
                If not specified, the builtin plugins will be used.

        Returns:
            dataset: A new dataset with specified contents
        """

        # TODO: remove the default value for media_type
        # https://github.com/openvinotoolkit/datumaro/issues/675

        if isinstance(categories, list):
            categories = {AnnotationType.label: LabelCategories.from_iterable(categories)}

        if not categories:
            categories = {}

        class _extractor(DatasetBase):
            def __init__(self):
                super().__init__(
                    length=len(iterable) if hasattr(iterable, "__len__") else None,
                    media_type=media_type,
                )

            def __iter__(self):
                return iter(iterable)

            def categories(self):
                return categories

        return cls.from_extractors(_extractor(), env=env)

    @staticmethod
    def from_extractors(*sources: IDataset, env: Optional[Environment] = None) -> Dataset:
        """
        Creates a new dataset from one or several `Extractor`s.

        In case of a single input, creates a lazy wrapper around the input.
        In case of several inputs, merges them and caches the resulting
        dataset.

        Parameters:
            sources: one or many input extractors
            env: A context for plugins, which will be used for this dataset.
                If not specified, the builtin plugins will be used.

        Returns:
            dataset: A new dataset with contents produced by input extractors
        """

        if len(sources) == 1:
            source = sources[0]
            dataset = Dataset(source=source, env=env)
        else:
            from datumaro.components.merge.exact_merge import ExactMerge

            media_type = ExactMerge.merge_media_types(sources)
            source = ExactMerge.merge(sources)
            categories = ExactMerge.merge_categories(s.categories() for s in sources)
            dataset = Dataset(source=source, categories=categories, media_type=media_type, env=env)
        return dataset

    def __init__(
        self,
        source: Optional[IDataset] = None,
        *,
        categories: Optional[CategoriesInfo] = None,
        media_type: Optional[Type[MediaElement]] = None,
        env: Optional[Environment] = None,
    ) -> None:
        super().__init__()

        assert env is None or isinstance(env, Environment), env
        self._env = env

        self.eager = None
        self._data = DatasetStorage(source, categories=categories, media_type=media_type)
        if self.is_eager:
            self.init_cache()

        self._format = DEFAULT_FORMAT
        self._source_path = None
        self._options = {}

    def define_categories(self, categories: CategoriesInfo) -> None:
        self._data.define_categories(categories)

    def init_cache(self) -> None:
        self._data.init_cache()

    def __iter__(self) -> Iterator[DatasetItem]:
        yield from self._data

    def __len__(self) -> int:
        return len(self._data)

    def get_subset(self, name) -> DatasetSubset:
        return DatasetSubset(self, name)

    def subsets(self) -> Dict[str, DatasetSubset]:
        return {k: self.get_subset(k) for k in self._data.subsets()}

    def infos(self) -> DatasetInfo:
        return {}

    def categories(self) -> CategoriesInfo:
        return self._data.categories()

    def media_type(self) -> Type[MediaElement]:
        return self._data.media_type()

    def ann_types(self) -> Set[AnnotationType]:
        return set()

    def get(self, id: str, subset: Optional[str] = None) -> Optional[DatasetItem]:
        return self._data.get(id, subset)

    def __contains__(self, x: Union[DatasetItem, str, Tuple[str, str]]) -> bool:
        if isinstance(x, DatasetItem):
            x = (x.id, x.subset)
        elif not isinstance(x, (tuple, list)):
            x = [x]
        return self.get(*x) is not None

    def put(
        self, item: DatasetItem, id: Optional[str] = None, subset: Optional[str] = None
    ) -> None:
        overrides = {}
        if id is not None:
            overrides["id"] = id
        if subset is not None:
            overrides["subset"] = subset
        if overrides:
            item = item.wrap(**overrides)

        self._data.put(item)

    def remove(self, id: str, subset: Optional[str] = None) -> None:
        self._data.remove(id, subset)

    def filter(
        self, expr: str, filter_annotations: bool = False, remove_empty: bool = False
    ) -> Dataset:
        """
        Filters out some dataset items or annotations, using a custom filter
        expression.

        Results are stored in-place. Modifications are applied lazily.

        Args:
            expr: XPath-formatted filter expression
                (e.g. `/item[subset = 'train']`,
                `/item/annotation[label = 'cat']`)
            filter_annotations: Indicates if the filter should be
                applied to items or annotations
            remove_empty: When filtering annotations, allows to
                exclude empty items from the resulting dataset

        Returns: self
        """

        if filter_annotations:
            return self.transform(XPathAnnotationsFilter, xpath=expr, remove_empty=remove_empty)
        else:
            return self.transform(XPathDatasetFilter, xpath=expr)

    def update(self, source: Union[DatasetPatch, IDataset, Iterable[DatasetItem]]) -> Dataset:
        """
        Updates items of the current dataset from another dataset or an
        iterable (the source). Items from the source overwrite matching
        items in the current dataset. Unmatched items are just appended.

        If the source is a DatasetPatch, the removed items in the patch
        will be removed in the current dataset.

        If the source is a dataset, labels are matched. If the labels match,
        but the order is different, the annotation labels will be remapped to
        the current dataset label order during updating.

        Returns: self
        """

        self._data.update(source)
        return self

    def transform(self, method: Union[str, Type[Transform]], **kwargs) -> Dataset:
        """
        Applies some function to dataset items.

        Results are stored in-place. Modifications are applied lazily.
        Transforms are not allowed to change media type of dataset items.

        Args:
            method: The transformation to be applied to the dataset.
                If a string is passed, it is treated as a plugin name,
                which is searched for in the dataset environment.
            **kwargs: Parameters for the transformation

        Returns: self
        """

        if isinstance(method, str):
            method = self.env.transforms[method]

        if not (inspect.isclass(method) and issubclass(method, Transform)):
            raise TypeError("Unexpected 'method' argument type: %s" % type(method))

        self._data.transform(method, **kwargs)
        if self.is_eager:
            self.init_cache()

        return self

    def run_model(
        self, model: Union[Launcher, Type[ModelTransform]], *, batch_size: int = 1, **kwargs
    ) -> Dataset:
        """
        Applies a model to dataset items' media and produces a dataset with
        media and annotations.

        Args:
            model: The model to be applied to the dataset
            batch_size: The number of dataset items processed
                simultaneously by the model
            **kwargs: Parameters for the model

        Returns: self
        """

        if isinstance(model, Launcher):
            return self.transform(ModelTransform, launcher=model, batch_size=batch_size, **kwargs)
        elif inspect.isclass(model) and isinstance(model, ModelTransform):
            return self.transform(model, batch_size=batch_size, **kwargs)
        else:
            raise TypeError("Unexpected 'model' argument type: %s" % type(model))

    def select(self, pred: Callable[[DatasetItem], bool]) -> Dataset:
        class _DatasetFilter(ItemTransform):
            def transform_item(self, item):
                if pred(item):
                    return item
                return None

        return self.transform(_DatasetFilter)

    @property
    def data_path(self) -> Optional[str]:
        return self._source_path

    @property
    def format(self) -> Optional[str]:
        return self._format

    @property
    def options(self) -> Dict[str, Any]:
        return self._options

    @property
    def is_modified(self) -> bool:
        return self._data.has_updated_items()

    def get_patch(self) -> DatasetPatch:
        return self._data.get_patch()

    @property
    def env(self) -> Environment:
        if not self._env:
            self._env = Environment()
        return self._env

    @property
    def is_cache_initialized(self) -> bool:
        return self._data.is_cache_initialized()

    @property
    def is_eager(self) -> bool:
        return self.eager if self.eager is not None else self._global_eager

    @property
    def is_bound(self) -> bool:
        return bool(self._source_path) and bool(self._format)

    def bind(
        self, path: str, format: Optional[str] = None, *, options: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Binds the dataset to a speific directory.
        Allows to set default saving parameters.

        The following saves will be done to this directory by default and will
        use the saved parameters.
        """

        self._source_path = path
        self._format = format or DEFAULT_FORMAT
        self._options = options or {}

    def flush_changes(self):
        self._data.flush_changes()

    @scoped
    def export(
        self,
        save_dir: str,
        format: Union[str, Type[Exporter]],
        *,
        progress_reporter: Optional[ProgressReporter] = None,
        error_policy: Optional[ExportErrorPolicy] = None,
        **kwargs,
    ) -> None:
        """
        Saves the dataset in some format.

        Args:
            save_dir: The output directory
            format: The desired output format.
                If a string is passed, it is treated as a plugin name,
                which is searched for in the dataset environment.
            progress_reporter: An object to report progress
            error_policy: An object to report format-related errors
            **kwargs: Parameters for the format
        """

        if not save_dir:
            raise ValueError("Dataset export path is not specified")

        inplace = save_dir == self._source_path and format == self._format

        if isinstance(format, str):
            converter = self.env.converters[format]
        else:
            converter = format

        if not (inspect.isclass(converter) and issubclass(converter, Exporter)):
            raise TypeError("Unexpected 'format' argument type: %s" % type(converter))

        save_dir = osp.abspath(save_dir)
        if not osp.exists(save_dir):
            on_error_do(rmtree, save_dir, ignore_errors=True)
            inplace = False
        os.makedirs(save_dir, exist_ok=True)

        has_ctx_args = progress_reporter is not None or error_policy is not None

        if not progress_reporter:
            progress_reporter = NullProgressReporter()

        assert "ctx" not in kwargs
        converter_kwargs = copy(kwargs)
        converter_kwargs["ctx"] = ExportContext(
            progress_reporter=progress_reporter, error_policy=error_policy
        )

        try:
            if not inplace:
                try:
                    converter.convert(self, save_dir=save_dir, **converter_kwargs)
                except TypeError as e:
                    # TODO: for backward compatibility. To be removed after 0.3
                    if "unexpected keyword argument 'ctx'" not in str(e):
                        raise

                    if has_ctx_args:
                        warnings.warn(
                            "It seems that '%s' converter "
                            "does not support progress and error reporting, "
                            "it will be disabled" % format,
                            DeprecationWarning,
                        )
                    converter_kwargs.pop("ctx")

                    converter.convert(self, save_dir=save_dir, **converter_kwargs)
            else:
                try:
                    converter.patch(self, self.get_patch(), save_dir=save_dir, **converter_kwargs)
                except TypeError as e:
                    # TODO: for backward compatibility. To be removed after 0.3
                    if "unexpected keyword argument 'ctx'" not in str(e):
                        raise

                    if has_ctx_args:
                        warnings.warn(
                            "It seems that '%s' converter "
                            "does not support progress and error reporting, "
                            "it will be disabled" % format,
                            DeprecationWarning,
                        )
                    converter_kwargs.pop("ctx")

                    converter.patch(self, self.get_patch(), save_dir=save_dir, **converter_kwargs)
        except _ExportFail as e:
            raise e.__cause__

        self.bind(save_dir, format, options=copy(kwargs))
        self.flush_changes()

    def save(self, save_dir: Optional[str] = None, **kwargs) -> None:
        options = dict(self._options)
        options.update(kwargs)

        self.export(save_dir or self._source_path, format=self._format, **options)

    @classmethod
    def load(cls, path: str, **kwargs) -> Dataset:
        return cls.import_from(path, format=DEFAULT_FORMAT, **kwargs)

    @classmethod
    def import_from(
        cls,
        path: str,
        format: Optional[str] = None,
        *,
        env: Optional[Environment] = None,
        progress_reporter: Optional[ProgressReporter] = None,
        error_policy: Optional[ImportErrorPolicy] = None,
        **kwargs,
    ) -> Dataset:
        """
        Creates a `Dataset` instance from a dataset on the disk.

        Args:
            path - The input file or directory path
            format - Dataset format.
                If a string is passed, it is treated as a plugin name,
                which is searched for in the `env` plugin context.
                If not set, will try to detect automatically,
                using the `env` plugin context.
            env - A plugin collection. If not set, the built-in plugins are used
            progress_reporter - An object to report progress.
                Implies earger loading.
            error_policy - An object to report format-related errors.
                Implies earger loading.
            **kwargs - Parameters for the format
        """

        if env is None:
            env = Environment()

        if not format:
            format = cls.detect(path, env=env)

        # TODO: remove importers, put this logic into extractors
        if format in env.importers:
            importer = env.make_importer(format)
            with logging_disabled(log.INFO):
                detected_sources = importer(path, **kwargs)
        elif format in env.extractors:
            detected_sources = [{"url": path, "format": format, "options": kwargs}]
        else:
            raise UnknownFormatError(format)

        # TODO: probably, should not be available in lazy mode, because it
        # becomes unreliable and error-prone. For progress reporting it
        # makes little sense, because loading stage is spread over other
        # operations. Error reporting is going to be unreliable.
        has_ctx_args = progress_reporter is not None or error_policy is not None
        eager = has_ctx_args

        if not progress_reporter:
            progress_reporter = NullProgressReporter()
        pbars = progress_reporter.split(len(detected_sources))

        try:
            extractors = []
            for src_conf, pbar in zip(detected_sources, pbars):
                if not isinstance(src_conf, Source):
                    src_conf = Source(src_conf)

                extractor_kwargs = dict(src_conf.options)

                assert "ctx" not in extractor_kwargs
                extractor_kwargs["ctx"] = ImportContext(
                    progress_reporter=pbar, error_policy=error_policy
                )

                try:
                    extractors.append(
                        env.make_extractor(src_conf.format, src_conf.url, **extractor_kwargs)
                    )
                except TypeError as e:
                    # TODO: for backward compatibility. To be removed after 0.3
                    if "unexpected keyword argument 'ctx'" not in str(e):
                        raise

                    if has_ctx_args:
                        warnings.warn(
                            "It seems that '%s' extractor "
                            "does not support progress and error reporting, "
                            "it will be disabled" % src_conf.format,
                            DeprecationWarning,
                        )
                    extractor_kwargs.pop("ctx")

                    extractors.append(
                        env.make_extractor(src_conf.format, src_conf.url, **extractor_kwargs)
                    )

            dataset = cls.from_extractors(*extractors, env=env)
            if eager:
                dataset.init_cache()
        except _ImportFail as e:
            raise e.__cause__

        dataset._source_path = path
        dataset._format = format

        return dataset

    @staticmethod
    def detect(path: str, *, env: Optional[Environment] = None, depth: int = 2) -> str:
        """
        Attempts to detect dataset format of a given directory.

        This function tries to detect a single format and fails if it's not
        possible. Check Environment.detect_dataset() for a function that
        reports status for each format checked.

        Args:
            path: The directory to check
            depth: The maximum depth for recursive search
            env: A plugin collection. If not set, the built-in plugins are used
        """

        if env is None:
            env = Environment()

        if depth < 0:
            raise ValueError("Depth cannot be less than zero")

        matches = env.detect_dataset(path, depth=depth)
        if not matches:
            raise NoMatchingFormatsError()
        elif 1 < len(matches):
            raise MultipleFormatsMatchError(matches)
        else:
            return matches[0]


@contextmanager
def eager_mode(new_mode: bool = True, dataset: Optional[Dataset] = None) -> None:
    if dataset is not None:
        old_mode = dataset.eager

        try:
            dataset.eager = new_mode
            yield
        finally:
            dataset.eager = old_mode
    else:
        old_mode = Dataset._global_eager

        try:
            Dataset._global_eager = new_mode
            yield
        finally:
            Dataset._global_eager = old_mode
