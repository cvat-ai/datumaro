# Copyright (C) 2023 Intel Corporation
#
# SPDX-License-Identifier: MIT

# ruff: noqa: E501

from collections import defaultdict
from typing import Dict, List

import numpy as np

from datumaro.components.annotation import FeatureVector
from datumaro.components.dataset import IDataset
from datumaro.components.launcher import LauncherWithModelInterpreter
from datumaro.util import take_by


class RunningStats1D:
    def __init__(self):
        self.running_mean = None
        self.running_sq_mean = None
        self.num: int = 0

    def add(self, feats: List[FeatureVector]) -> None:
        arr = np.stack([feat.vector for feat in feats], axis=0)
        assert arr.ndim == 2

        batch_size, _ = arr.shape
        mean = arr.mean(0)
        arr = np.expand_dims(arr, axis=-1)  # B x D x 1
        sq_mean = np.mean(np.matmul(arr, np.transpose(arr, axes=(0, 2, 1))), axis=0)  # D x D

        self.num += batch_size

        if self.running_mean is not None:
            self.running_mean = self.running_mean + batch_size / float(self.num) * (
                mean - self.running_mean
            )
        else:
            self.running_mean = mean

        if self.running_sq_mean is not None:
            self.running_sq_mean = self.running_sq_mean + batch_size / float(self.num) * (
                sq_mean - self.running_sq_mean
            )
        else:
            self.running_sq_mean = sq_mean

    @property
    def mean(self) -> np.ndarray:
        return self.running_mean

    @property
    def cov(self) -> np.ndarray:
        mean = np.expand_dims(self.running_mean, axis=-1)  # D x 1
        return self.running_sq_mean - np.matmul(mean, mean.transpose())


class FeatureAccumulator:
    def __init__(self, model: LauncherWithModelInterpreter):
        self.model = model
        self._batch_size = 1

    def get_activation_stats(self, dataset: IDataset) -> RunningStats1D:
        running_stats = RunningStats1D()

        for batch in take_by(dataset, self._batch_size):
            outputs = self.model.launch(batch)[0]
            features = [outputs[-1]]  # extracted feature vector of googlenet-v4
            running_stats.add(features)

        return running_stats


class FeatureAccumulatorByLabel(FeatureAccumulator):
    def __init__(self, model):
        super().__init__(model)

    def get_activation_stats(self, dataset: IDataset) -> Dict[int, RunningStats1D]:
        running_stats = defaultdict(RunningStats1D)

        for batch in take_by(dataset, self._batch_size):
            inputs, targets = [], []
            for item in batch:
                for ann in item.annotations:
                    inputs.append(np.atleast_3d(item.media.data))
                    targets.append(ann.label)

            outputs = self.model.launch(batch)[0]
            features = [outputs[-1]]  # extracted feature vector of googlenet-v4

            for target in targets:
                running_stats[target].add(features)

        return running_stats
