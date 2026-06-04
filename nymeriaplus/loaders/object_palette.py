# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Shared category color palette for object annotations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CategoryPaletteEntry:
    category: str
    category_uid: int
    color_u8: tuple[int, int, int]

    @property
    def color_float(self) -> tuple[float, float, float]:
        return tuple(v / 255.0 for v in self.color_u8)


def generate_category_color(category_uid: int) -> np.ndarray:
    """Generate the same category color as the Nymeria dataset viewer."""
    hue = (int(category_uid) * 30) % 360
    if hue < 60:
        color = [255, int(hue * 4.25), 0]
    elif hue < 120:
        color = [255 - int((hue - 60) * 4.25), 255, 0]
    elif hue < 180:
        color = [0, 255, int((hue - 120) * 4.25)]
    elif hue < 240:
        color = [0, 255 - int((hue - 180) * 4.25), 255]
    elif hue < 300:
        color = [int((hue - 240) * 4.25), 0, 255]
    else:
        color = [255, 0, 255 - int((hue - 300) * 4.25)]
    return np.array(color, dtype=np.uint8)


def build_category_palette(
    instances: dict[str, dict],
) -> dict[str, CategoryPaletteEntry]:
    palette: dict[str, CategoryPaletteEntry] = {}
    for instance in instances.values():
        category = str(instance.get("category", "Unknown"))
        category_uid = int(instance.get("category_uid", 0))
        if category in palette:
            continue
        color = generate_category_color(category_uid)
        palette[category] = CategoryPaletteEntry(
            category=category,
            category_uid=category_uid,
            color_u8=tuple(int(v) for v in color),
        )
    return palette
