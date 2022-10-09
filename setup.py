# Copyright (C) 2019-2022 Intel Corporation
#
# SPDX-License-Identifier: MIT

import os
from distutils.util import strtobool

import setuptools

CORE_REQUIREMENTS_FILE = "requirements-core.txt"
DEFAULT_REQUIREMENTS_FILE = "requirements-default.txt"


def parse_requirements(filename=CORE_REQUIREMENTS_FILE):
    with open(filename) as fh:
        return fh.readlines()


CORE_REQUIREMENTS = parse_requirements(CORE_REQUIREMENTS_FILE)
if strtobool(os.getenv("DATUMARO_HEADLESS", "0").lower()):
    CORE_REQUIREMENTS.append("opencv-python-headless")
else:
    CORE_REQUIREMENTS.append("opencv-python")

DEFAULT_REQUIREMENTS = parse_requirements(DEFAULT_REQUIREMENTS_FILE)

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="datumaro",
    version="0.3",
    author="Intel",
    author_email="maxim.zhiltsov@intel.com",
    description="Dataset Management Framework (Datumaro)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cvat-ai/datumaro",
    packages=setuptools.find_packages(include=["datumaro*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "tf": ["tensorflow"],
        "tfds": [
            "tensorflow-datasets!=4.5.0,!=4.5.1"
        ],  # 4.5.0 fails on Windows, https://github.com/tensorflow/datasets/issues/3709
        "tf-gpu": ["tensorflow-gpu"],
        "default": DEFAULT_REQUIREMENTS,
    },
    entry_points={
        "console_scripts": [
            "datum=datumaro.cli.__main__:main",
        ],
    },
    include_package_data=True,
)
