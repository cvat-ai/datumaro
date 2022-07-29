# Copyright (C) 2020-2021 Intel Corporation
# Copyright (C) 2022 CVAT.ai Corporation
#
# SPDX-License-Identifier: MIT

import argparse
import logging as log
from typing import List, Type

from datumaro.cli.util import MultilineFormatter
from datumaro.util import to_snake_case

_plugin_bases = None


def plugin_bases() -> List[Type["CliPlugin"]]:
    global _plugin_bases
    if _plugin_bases is None:
        from datumaro.components.converter import Converter
        from datumaro.components.extractor import Extractor, Importer, Transform
        from datumaro.components.launcher import Launcher
        from datumaro.components.validator import Validator

        _plugin_bases = [Launcher, Extractor, Transform, Importer, Converter, Validator]

    return _plugin_bases


class PluginNameBuilder:
    @staticmethod
    def _remove_plugin_type(name: str) -> str:
        for t in {"transform", "extractor", "converter", "launcher", "importer", "validator"}:
            name = name.replace("_" + t, "")
        return name

    def _sanitize_plugin_name(cls, name: str) -> str:
        return cls._remove_plugin_type(to_snake_case(name))

    def __get__(self, obj, objtype=None):
        if not objtype:
            objtype = type(obj)
        return self._sanitize_plugin_name(objtype.__name__)


class CliPlugin:
    NAME = PluginNameBuilder()

    @staticmethod
    def _get_doc(cls):
        doc = getattr(cls, "__doc__", "")
        if doc:
            if any(getattr(t, "__doc__", "") == doc for t in plugin_bases()):
                doc = ""
        return doc

    @classmethod
    def build_cmdline_parser(cls, **kwargs):
        args = {
            "prog": cls.NAME,
            "description": cls._get_doc(cls),
            "formatter_class": MultilineFormatter,
        }
        args.update(kwargs)

        return argparse.ArgumentParser(**args)

    @classmethod
    def parse_cmdline(cls, args=None):
        if args and args[0] == "--":
            args = args[1:]
        parser = cls.build_cmdline_parser()
        args = parser.parse_args(args)
        args = vars(args)

        log.debug(
            "Parsed parameters: \n\t%s", "\n\t".join("%s: %s" % (k, v) for k, v in args.items())
        )

        return args
