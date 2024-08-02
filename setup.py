# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="nymeria",
    version="0.0.1",
    packages=find_packages(),
    author="Lingni Ma",
    author_email="lingni.ma@meta.com",
    description="The official repo to support the Nymeria dataset",
    python_requires=">=3.10",
    install_requires=["click", "requests", "tqdm"],
)
