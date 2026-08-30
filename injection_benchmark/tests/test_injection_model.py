# fmt: off

"""Numerical checks for the injection fluence-S/N normalization."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generate_injections import FLUENCE_SNR_RANGE, sample_parameter_rows
from injection_model import scattered_gaussian_profile, template_l2_norm


class FluenceSnrNormalizationTest(unittest.TestCase):
    def test_sampled_fluence_snr_stays_in_requested_range(self) -> None:
        rows = sample_parameter_rows(
            256,
            np.random.default_rng(20260613),
            scattering_ref_freq_mhz=1500.0,
        )
        values = np.asarray([row["fluence_snr"] for row in rows])
        self.assertGreaterEqual(float(values.min()), FLUENCE_SNR_RANGE[0])
        self.assertLessEqual(float(values.max()), FLUENCE_SNR_RANGE[1])

    def test_discrete_template_reaches_requested_fluence_snr(self) -> None:
        channel_count = 64
        times         = np.arange(-200.0, 1200.0)
        centers       = np.full(channel_count, 0.37)
        sigma_samples = 8.6
        tau_samples   = np.linspace(0.0, 40.0, channel_count)
        frequencies   = np.linspace(-2.0, 2.0, channel_count)
        envelope      = np.exp(-0.5 * np.square(frequencies / 0.8))

        pulse = scattered_gaussian_profile(
            times,
            centers,
            sigma_samples,
            tau_samples,
        )
        norm = template_l2_norm(envelope, sigma_samples, tau_samples)

        for requested_snr in (3.0, 100.0):
            whitened_signal = requested_snr / norm * envelope[:, None] * pulse
            measured_snr    = float(np.sqrt(np.sum(np.square(whitened_signal))))
            self.assertAlmostEqual(
                measured_snr,
                requested_snr,
                delta=requested_snr * 1e-6,
            )


if __name__ == "__main__":
    unittest.main()

# fmt: on
