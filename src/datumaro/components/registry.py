# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

from typing import Callable, Dict, Generic, Iterable, Iterator, Optional, Type, TypeVar

from datumaro.components.cli_plugin import CliPlugin

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self):
        self.items: Dict[str, T] = {}

    def register(self, name: str, value: T) -> T:
        self.items[name] = value
        return value

    def unregister(self, name: str) -> Optional[T]:
        return self.items.pop(name, None)

    def get(self, key: str):
        """Returns a class or a factory function"""
        return self.items[key]

    def __getitem__(self, key: str) -> T:
        return self.get(key)

    def __contains__(self, key) -> bool:
        return key in self.items

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)


class PluginRegistry(Registry[Type[CliPlugin]]):
    def __init__(
        self, filter: Callable[[Type[CliPlugin]], bool] = None
    ):  # pylint: disable=redefined-builtin
        super().__init__()
        self._filter = filter

    def batch_register(self, values: Iterable[CliPlugin]):
        for v in values:
            if self._filter and not self._filter(v):
                continue

            self.register(v.NAME, v)
