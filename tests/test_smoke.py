# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from nymeriaplus.layout import BodyModel, DataGroups, get_group_definitions


def test_body_model_values() -> None:
    assert {m.value for m in BodyModel} == {"smpl", "mhr"}


def test_data_groups_definitions_nonempty() -> None:
    defs = get_group_definitions()
    for dg in DataGroups:
        assert dg.value in defs
        assert isinstance(defs[dg.value], list)
        assert defs[dg.value], f"empty group {dg.value}"
