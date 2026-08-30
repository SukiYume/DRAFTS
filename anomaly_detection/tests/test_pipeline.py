# fmt: off

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from dataset_h5 import DataAugmentation, H5PairDataset, H5RadioDataset
from models import (
    RadioAutoencoder,
    VariationalRadioAutoencoder,
    compute_anomaly_scores,
)
from prepare_data import (
    _collect_fits_files,
    preprocess_slice,
    processing_fits_to_slices,
)
from run_inference import compute_threshold
from train_ddp import build_indices


def test_augmentation_uses_time_and_frequency_axes(monkeypatch):
    data = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)

    def frequency_mask(size=None):
        assert size == 4
        return np.array([0.0, 1.0, 0.0, 1.0])

    monkeypatch.setattr(np.random, "random", frequency_mask)
    augmented = DataAugmentation(
        flip_time      = False,
        flip_freq      = False,
        noise_std      = 0,
        time_shift     = 0,
        freq_mask_prob = 0.5,
    )(data)
    expected            = data.copy()
    expected[:, [0, 2]] = 0
    np.testing.assert_array_equal(augmented, expected)

    monkeypatch.setattr(np.random, "randint", lambda *_: 1)
    shifted = DataAugmentation(
        flip_time      = False,
        flip_freq      = False,
        noise_std      = 0,
        time_shift     = 0.5,
        freq_mask_prob = 0,
    )(data)
    np.testing.assert_array_equal(shifted, np.roll(data, 1, axis=0))


def test_preprocessing_handles_constant_nonfinite_and_short_factors():
    constant       = np.full((512, 512), 7.0, dtype=np.float32)
    constant[0, 0] = np.nan
    normalized     = preprocess_slice(constant)
    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert not normalized.any()

    data   = np.arange(1024 * 512, dtype=np.float32).reshape(1024, 512)
    slices = processing_fits_to_slices(data, [1, 4])
    assert [factor for factor, _ in slices] == [1]
    assert slices[0][1].shape == (2, 512, 512)
    assert 0 <= slices[0][1].min() <= slices[0][1].max() <= 1


def test_file_collection_has_no_hidden_beam_name_filter(tmp_path):
    generic_fits = tmp_path / "observation.fits"
    upper_fits   = tmp_path / "beam.FIT"
    ignored      = tmp_path / "notes.txt"
    for path in (generic_fits, upper_fits, ignored):
        path.touch()

    assert _collect_fits_files(tmp_path) == sorted(
        [str(generic_fits.resolve()), str(upper_fits.resolve())]
    )


def test_pair_dataset_never_crosses_sources_or_slice_gaps(tmp_path):
    h5_path       = tmp_path / "pairs.h5"
    names         = ["a.fits", "b.fits", "a.fits", "a.fits", "b.fits", "c.fits"]
    factors       = [8, 8, 8, 8, 8, 8]
    slice_indices = [1, 0, 0, 3, 1, 0]
    with h5py.File(h5_path, "w") as h5:
        h5.create_dataset("images", data=np.zeros((6, 512, 512), dtype=np.float32))
        h5.create_dataset(
            "names",
            data  = np.asarray(names, dtype=object).reshape(-1, 1),
            dtype = h5py.string_dtype("utf-8"),
        )
        h5.create_dataset("down_time", data=np.asarray(factors).reshape(-1, 1))
        h5.create_dataset("slice_idx", data=np.asarray(slice_indices).reshape(-1, 1))

    pairs = H5PairDataset(str(h5_path))
    assert pairs.pairs == [(2, 0), (1, 4)]
    previous, current = pairs[0]
    assert previous.shape == current.shape == (1, 512, 512)

    image, metadata = H5RadioDataset(
        str(h5_path), indices=np.array([4]), return_meta=True
    )[0]
    assert image.shape == (1, 512, 512)
    assert metadata == {"name": "b.fits", "down_time": 8, "slice_idx": 1}

    train_indices, validation_indices = build_indices(
        str(h5_path), val_ratio=0.5, seed=3
    )
    train_names      = {names[index] for index in train_indices}
    validation_names = {names[index] for index in validation_indices}
    assert train_names.isdisjoint(validation_names)


def test_anomaly_score_and_threshold_contract():
    target         = torch.zeros(2, 1, 2, 2)
    reconstruction = torch.ones_like(target)
    mu             = torch.ones(2, 3, 1, 1)
    logvar         = torch.zeros_like(mu)
    total, recon_score, kl_score = compute_anomaly_scores(
        reconstruction,
        target,
        mu,
        logvar,
        beta=0.2,
    )
    torch.testing.assert_close(recon_score, torch.full((2,), 0.5))
    torch.testing.assert_close(kl_score, torch.full((2,), 0.5))
    torch.testing.assert_close(total, torch.full((2,), 0.6))
    assert compute_threshold([1.0, 2.0, 3.0], "percentile") == pytest.approx(2.9)
    with pytest.raises(ValueError, match="finite"):
        compute_threshold([np.nan])


def test_models_preserve_shape_and_vae_eval_is_deterministic():
    inputs = torch.rand(1, 1, 64, 64)
    for encoder in ("convnext", "resnet18"):
        autoencoder = RadioAutoencoder(pretrained=False, encoder_name=encoder).eval()
        with torch.no_grad():
            assert autoencoder(inputs).shape == inputs.shape

        vae = VariationalRadioAutoencoder(
            pretrained   = False,
            latent_ch    = 8,
            encoder_name = encoder,
        ).eval()
        with torch.no_grad():
            first  = vae(inputs)[0]
            second = vae(inputs)[0]
        assert first.shape == inputs.shape
        torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_retained_checkpoint_matches_documented_model():
    checkpoint_path = Path(__file__).parents[1] / "best_loss_model.pth"
    if not checkpoint_path.is_file():
        pytest.skip("retained local checkpoint is not present")

    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = VariationalRadioAutoencoder(
        pretrained   = False,
        latent_ch    = 128,
        encoder_name = "convnext",
    )
    model.load_state_dict(state, strict=True)

# fmt: on
