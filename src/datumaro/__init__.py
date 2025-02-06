# Copyright (C) 2019-2022 Intel Corporation
# Copyright (C) 2022-2024 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

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
from .components.contexts.importer import FailingImportErrorPolicy, ImportErrorPolicy
from .components.dataset import Dataset, DatasetSubset, IDataset, eager_mode
from .components.dataset_base import CategoriesInfo, DatasetBase, DatasetItem, SubsetBase
from .components.dataset_item_storage import ItemStatus
from .components.dataset_storage import DatasetPatch
from .components.environment import Environment
from .components.exporter import Exporter, ExportErrorPolicy, FailingExportErrorPolicy
from .components.hl_ops import (  # pylint: disable=redefined-builtin
    export,
    filter,
    merge,
    run_model,
    transform,
    validate,
)
from .components.importer import Importer
from .components.launcher import Launcher
from .components.media import Image, MediaElement, PointCloud, Video, VideoFrame
from .components.media_manager import MediaManager
from .components.progress_reporting import NullProgressReporter, ProgressReporter
from .components.registry import PluginRegistry
from .components.transformer import ItemTransform, ModelTransform, Transform
from .components.validator import Validator
from .util.definitions import DEFAULT_SUBSET_NAME
from .version import __version__
