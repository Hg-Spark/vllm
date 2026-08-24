# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.v1.attention.selector import _parse_pcp_latent_mla_config


@pytest.mark.parametrize(
    ("additional_config", "expected"),
    [
        (None, (False, False)),
        ({}, (False, False)),
        ({"pcp_latent_mla": False}, (False, False)),
        ({"pcp_latent_mla": True}, (True, False)),
        ({"pcp_latent_mla": {}}, (True, False)),
        ({"pcp_latent_mla": {"enabled": True}}, (True, False)),
        (
            {"pcp_latent_mla": {"enabled": True, "strict": True}},
            (True, True),
        ),
    ],
)
def test_parse_pcp_latent_mla_config(additional_config, expected):
    assert _parse_pcp_latent_mla_config(additional_config) == expected


@pytest.mark.parametrize(
    "additional_config",
    [
        {"pcp_latent_mla": "yes"},
        {"pcp_latent_mla": {"enabled": 1}},
        {"pcp_latent_mla": {"strict": 1}},
        {"pcp_latent_mla": {"enabled": False, "strict": True}},
        {"pcp_latent_mla": {"unknown": True}},
    ],
)
def test_parse_pcp_latent_mla_config_rejects_invalid(additional_config):
    with pytest.raises(ValueError):
        _parse_pcp_latent_mla_config(additional_config)
