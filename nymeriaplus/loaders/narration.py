# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Loader for narration CSVs. Not implemented yet."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class NarrationLoader:
    """Stub. Will load motion / atomic action / activity summarization CSVs."""

    def __init__(self, narration_dir: Path) -> None:
        self.narration_dir = narration_dir
        logger.warning("NarrationLoader is not implemented yet")

    @property
    def is_valid(self) -> bool:
        return False
