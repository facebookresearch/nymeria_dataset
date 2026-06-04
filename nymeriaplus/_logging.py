# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import sys


def configure(level: int | str = logging.INFO) -> None:
    """Configure root logging for CLI usage. Idempotent."""
    root = logging.getLogger()
    if any(getattr(h, "_nymeriaplus", False) for h in root.handlers):
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(levelname)-7s %(name)s:%(funcName)s:%(lineno)d %(message)s"
        )
    )
    handler._nymeriaplus = True  # type: ignore[attr-defined]
    root.addHandler(handler)
    root.setLevel(level)
