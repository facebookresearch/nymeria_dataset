# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from nymeriaplus.loaders.boxy import BoxyBBLoader
from nymeriaplus.loaders.mhr import MHRBodyLoader
from nymeriaplus.loaders.narration import NarrationLoader
from nymeriaplus.loaders.recording import RecordingLoader
from nymeriaplus.loaders.shaper import ShaperMeshLoader
from nymeriaplus.loaders.smpl import SMPLBodyLoader
from nymeriaplus.loaders.xsens import XSensBodyLoader

__all__ = [
    "BoxyBBLoader",
    "MHRBodyLoader",
    "NarrationLoader",
    "RecordingLoader",
    "SMPLBodyLoader",
    "ShaperMeshLoader",
    "XSensBodyLoader",
]
