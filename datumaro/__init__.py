# Copyright (C) 2019-2022 Intel Corporation
# Copyright (C) 2022 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

# Borrowed from https://gist.github.com/benkehoe/066a73903e84576a8d6d911cfedc2df6
# With importlib.metadata the version can be obtained with just importlib.metadata.version
# This variable now just serves for backward compatibility and established practices.
try:
    # importlib.metadata is present in Python 3.8 and later
    import importlib.metadata as importlib_metadata
except ImportError:
    # use the shim package importlib-metadata pre-3.8
    import importlib_metadata as importlib_metadata

try:
    # __package__ allows for the case where __name__ is "__main__"
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    __version__ = "0.0.0"
version = __version__


from . import errors as errors
from . import ops as ops
from . import project as project
from .components.annotation import (
    NO_GROUP,
    Annotation,
    AnnotationType,
    Bbox,
    BinaryMaskImage,
    Caption,
    Categories,
    Colormap,
    CompiledMask,
    CompiledMaskImage,
    Cuboid3d,
    IndexMaskImage,
    Label,
    LabelCategories,
    Mask,
    MaskCategories,
    Points,
    PointsCategories,
    Polygon,
    PolyLine,
    RgbColor,
    RleMask,
    Skeleton,
)
from .components.cli_plugin import CliPlugin
from .components.converter import Converter, ExportErrorPolicy, FailingExportErrorPolicy
from .components.dataset import (
    Dataset,
    DatasetPatch,
    DatasetSubset,
    IDataset,
    ItemStatus,
    eager_mode,
)
from .components.environment import Environment, PluginRegistry
from .components.extractor import (
    DEFAULT_SUBSET_NAME,
    CategoriesInfo,
    DatasetItem,
    Extractor,
    FailingImportErrorPolicy,
    IExtractor,
    Importer,
    ImportErrorPolicy,
    ItemTransform,
    SourceExtractor,
    Transform,
)
from .components.hl_ops import (  # pylint: disable=redefined-builtin
    export,
    filter,
    merge,
    run_model,
    transform,
    validate,
)
from .components.launcher import Launcher, ModelTransform
from .components.media import ByteImage, Image, MediaElement, Video, VideoFrame
from .components.media_manager import MediaManager
from .components.progress_reporting import NullProgressReporter, ProgressReporter
from .components.validator import Validator
