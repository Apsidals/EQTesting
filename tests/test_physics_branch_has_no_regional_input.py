"""Enforces CLAUDE.md hard rule #5: the Universal Physics Encoder sees no
regional information. The forward signature must accept only a waveform
tensor, and perturbing site features / region metadata must not change
its embedding output (the encoder never touches them)."""

from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")

from eqxfer.config import ModelConfig
from eqxfer.data.geological import SITE_FEATURE_DIM
from eqxfer.models.split_transfer import (
    RegionalSiteEncoder,
    SplitTransferModel,
    UniversalPhysicsEncoder,
)


def _make_model() -> SplitTransferModel:
    torch.manual_seed(0)
    return SplitTransferModel(ModelConfig())


def test_universal_encoder_forward_signature_waveform_only() -> None:
    sig = inspect.signature(UniversalPhysicsEncoder.forward)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    assert len(params) == 1, (
        f"UniversalPhysicsEncoder.forward must take exactly one non-self "
        f"parameter (the waveform). Got: {[p.name for p in params]}"
    )
    assert params[0].name in {"waveform", "x"}, params[0].name


def test_universal_encoder_module_has_no_regional_submodules() -> None:
    """Names of submodules inside the universal encoder must not include
    site/region/vs30/lat/lon/station — a structural guard against someone
    wiring in leaking inputs later."""
    enc = UniversalPhysicsEncoder(ModelConfig())
    forbidden = ("site", "region", "vs30", "vp_crust", "lat", "lon", "station", "receiver")
    for name, _ in enc.named_modules():
        lower = name.lower()
        for token in forbidden:
            assert token not in lower, f"forbidden submodule name: {name!r}"


def test_perturbing_site_features_does_not_change_physics_embedding() -> None:
    """The SplitTransferModel has both branches; perturbing the site-feature
    input must leave the universal embedding bit-for-bit identical."""
    model = _make_model()
    model.eval()
    waveform = torch.randn(4, 3, 500)
    site_a = torch.randn(4, SITE_FEATURE_DIM)
    site_b = torch.randn(4, SITE_FEATURE_DIM)
    with torch.no_grad():
        emb_a = model.encode_physics(waveform)
        emb_b = model.encode_physics(waveform)  # deterministic
        # And cross-check: running the full forward with different site feats
        # must produce the same physics embed.
        out_a = model(waveform, site_a, return_embeddings=True)
        out_b = model(waveform, site_b, return_embeddings=True)
    assert torch.allclose(emb_a, emb_b)
    assert torch.allclose(out_a.physics_embed, out_b.physics_embed), (
        "Universal embedding changed when site features changed — the physics "
        "branch is leaking the site input somewhere."
    )
    # And the two site embeddings SHOULD differ, because the site branch does
    # take site features.
    assert not torch.allclose(out_a.site_embed, out_b.site_embed)


def test_regional_site_encoder_rejects_waveform_shaped_input() -> None:
    """Symmetric guard: the site encoder should not silently accept a
    waveform-shaped tensor."""
    enc = RegionalSiteEncoder(ModelConfig())
    waveform = torch.randn(4, 3, 500)
    with pytest.raises(ValueError):
        enc(waveform)


def test_universal_encoder_rejects_non_three_channel_input() -> None:
    enc = UniversalPhysicsEncoder(ModelConfig())
    wrong = torch.randn(4, 2, 500)  # 2 channels
    with pytest.raises(ValueError):
        enc(wrong)
