"""Tests for the EQ-HBC channel simulator."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import signal as sp_signal

from openbdn.channel.eq_hbc import EQChannelSimulator


def _random_bits(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 2, size=n, dtype=np.uint8)


# ---------------------------------------------------------------- line code
@pytest.mark.parametrize("line_code", ["manchester", "nrz"])
def test_modulate_demodulate_roundtrip(line_code: str) -> None:
    ch = EQChannelSimulator(samples_per_symbol=4, line_code=line_code)  # type: ignore[arg-type]
    bits = _random_bits(257, seed=0)
    wave = ch.modulate(bits)
    assert wave.shape == (257 * 4,)
    assert set(np.unique(wave)) == {-1.0, 1.0}
    np.testing.assert_array_equal(ch.demodulate(wave), bits)


def test_manchester_waveform_is_dc_free_per_bit() -> None:
    ch = EQChannelSimulator(samples_per_symbol=4)
    np.testing.assert_array_equal(ch.modulate([1, 0]), [1, 1, -1, -1, -1, -1, 1, 1])
    wave = ch.modulate(_random_bits(64, seed=1))
    np.testing.assert_array_equal(wave.reshape(-1, 4).sum(axis=1), 0.0)


def test_manchester_detector_ignores_dc_offset() -> None:
    ch = EQChannelSimulator()
    bits = _random_bits(100, seed=2)
    np.testing.assert_array_equal(ch.demodulate(ch.modulate(bits) + 7.5), bits)


def test_modulate_respects_custom_levels() -> None:
    ch = EQChannelSimulator(samples_per_symbol=2, levels=(0.0, 3.3), line_code="nrz")
    np.testing.assert_allclose(ch.modulate([0, 1, 1]), [0, 0, 3.3, 3.3, 3.3, 3.3])
    np.testing.assert_array_equal(ch.demodulate([0.2, 0.1, 3.0, 2.9]), [0, 1])


def test_modulate_rejects_non_binary() -> None:
    with pytest.raises(ValueError):
        EQChannelSimulator().modulate([0, 1, 2])


def test_demodulate_rejects_partial_symbol() -> None:
    with pytest.raises(ValueError):
        EQChannelSimulator(samples_per_symbol=4).demodulate(np.ones(6))


# -------------------------------------------------------------- noise model
@pytest.mark.parametrize("snr_db", [-5.0, 0.0, 6.0, 20.0])
def test_noise_hits_requested_snr(snr_db: float) -> None:
    ch = EQChannelSimulator(seed=1)
    clean = ch.modulate(_random_bits(2000, seed=1))
    noisy = ch.add_muscle_noise(clean, snr_db)
    assert noisy.shape == clean.shape
    assert ch.snr_db(clean, noisy) == pytest.approx(snr_db, abs=1e-9)


def test_noise_does_not_modify_input() -> None:
    ch = EQChannelSimulator(seed=2)
    clean = ch.modulate(_random_bits(100, seed=2))
    snapshot = clean.copy()
    ch.add_muscle_noise(clean, 3.0)
    np.testing.assert_array_equal(clean, snapshot)


def test_noise_is_reproducible_with_seed() -> None:
    clean = EQChannelSimulator().modulate(_random_bits(500, seed=3))
    a = EQChannelSimulator(seed=42).add_muscle_noise(clean, 5.0)
    b = EQChannelSimulator(seed=42).add_muscle_noise(clean, 5.0)
    c = EQChannelSimulator(seed=43).add_muscle_noise(clean, 5.0)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_is_coloured_like_emg() -> None:
    """Noise power must sit in the EMG band, not be spread like white noise."""
    ch = EQChannelSimulator(sample_rate_hz=8000.0, seed=4)
    clean = ch.modulate(_random_bits(8000, seed=4))
    noise = ch.add_muscle_noise(clean, 0.0) - clean
    freqs, psd = sp_signal.welch(noise, fs=ch.sample_rate_hz, nperseg=2048)

    in_band = psd[(freqs >= 10) & (freqs <= 500)].sum()
    assert in_band / psd.sum() > 0.85  # white noise would give ~0.12
    peak = freqs[np.argmax(psd)]
    assert 20.0 <= peak <= 200.0
    assert psd[0] < 0.2 * psd.max()  # spectral null towards DC


def test_noise_is_nonstationary() -> None:
    """Movement bursts make short-time noise power vary far more than chance."""
    ch = EQChannelSimulator(seed=5, burst_floor=0.1, burst_rate_hz=4.0)
    clean = ch.modulate(_random_bits(16000, seed=5))
    noise = ch.add_muscle_noise(clean, 0.0) - clean
    power = np.square(noise).reshape(-1, 800).mean(axis=1)
    assert power.max() / power.min() > 5.0


def test_emg_spectrum_shape() -> None:
    ch = EQChannelSimulator()
    spectrum = ch.emg_power_spectrum([0.0, 80.0, 3000.0])
    assert spectrum[0] == 0.0
    assert spectrum[1] > spectrum[2] > 0.0


@pytest.mark.parametrize(
    "bad", [np.zeros(0), np.zeros(16), np.ones((4, 4)), np.array([1.0, np.nan])]
)
def test_noise_rejects_invalid_signals(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        EQChannelSimulator(seed=0).add_muscle_noise(bad, 5.0)


# -------------------------------------------------------------- NLMS filter
@pytest.mark.parametrize(("line_code", "min_gain_db"), [("manchester", 6.0), ("nrz", 1.0)])
@pytest.mark.parametrize("snr_db", [0.0, 5.0, 10.0])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nlms_improves_snr_decision_directed(
    line_code: str, min_gain_db: float, snr_db: float, seed: int
) -> None:
    ch = EQChannelSimulator(seed=seed, line_code=line_code)  # type: ignore[arg-type]
    bits = _random_bits(4000, seed=seed)
    clean = ch.modulate(bits)
    noisy = ch.add_muscle_noise(clean, snr_db)
    filtered = ch.apply_adaptive_filter(noisy)

    assert filtered.shape == noisy.shape
    snr_in = ch.snr_db(clean, noisy)
    snr_out = ch.snr_db(clean, filtered)
    assert snr_out > snr_in + min_gain_db, f"gain was only {snr_out - snr_in:.2f} dB"


@pytest.mark.parametrize(("line_code", "snr_db"), [("nrz", 3.0), ("manchester", -18.0)])
def test_nlms_reduces_bit_errors(line_code: str, snr_db: float) -> None:
    errors_before = errors_after = 0
    for seed in range(4):
        ch = EQChannelSimulator(seed=seed, line_code=line_code)  # type: ignore[arg-type]
        bits = _random_bits(4000, seed=seed)
        clean = ch.modulate(bits)
        noisy = ch.add_muscle_noise(clean, snr_db)
        filtered = ch.apply_adaptive_filter(noisy)
        errors_before += int(np.sum(ch.demodulate(noisy) != bits))
        errors_after += int(np.sum(ch.demodulate(filtered) != bits))
    assert errors_before > 0
    assert errors_after < 0.6 * errors_before


def test_nlms_training_preamble_helps_convergence() -> None:
    ch = EQChannelSimulator(seed=7, line_code="nrz")
    bits = _random_bits(1500, seed=7)
    clean = ch.modulate(bits)
    noisy = ch.add_muscle_noise(clean, 2.0)
    n_train = 2000
    trained = ch.apply_adaptive_filter(noisy, training=clean[:n_train])
    tail = slice(n_train, None)
    assert ch.snr_db(clean[tail], trained[tail]) > ch.snr_db(clean[tail], noisy[tail]) + 2.0


def test_nlms_is_transparent_on_clean_signal() -> None:
    ch = EQChannelSimulator()
    clean = ch.modulate(_random_bits(500, seed=8))
    out = ch.apply_adaptive_filter(clean)
    np.testing.assert_allclose(out, clean, atol=1e-9)
    assert ch.taps is not None and ch.taps.shape == (ch.num_taps,)


def test_nlms_taps_stay_finite_under_heavy_noise() -> None:
    ch = EQChannelSimulator(seed=9, step_size=1.0)
    clean = ch.modulate(_random_bits(2000, seed=9))
    out = ch.apply_adaptive_filter(ch.add_muscle_noise(clean, -10.0))
    assert np.isfinite(out).all()
    assert ch.taps is not None and np.isfinite(ch.taps).all()


def test_nlms_rejects_overlong_training() -> None:
    ch = EQChannelSimulator()
    with pytest.raises(ValueError):
        ch.apply_adaptive_filter(np.ones(8), training=np.ones(9))


# ------------------------------------------------------------- construction
@pytest.mark.parametrize(
    "kwargs",
    [
        {"sample_rate_hz": 0},
        {"samples_per_symbol": 0},
        {"samples_per_symbol": 3},
        {"line_code": "ami"},
        {"num_taps": 0},
        {"step_size": 0.0},
        {"step_size": 2.0},
        {"regularization": 0.0},
        {"levels": (1.0, -1.0)},
        {"emg_corner_low_hz": 200.0, "emg_corner_high_hz": 100.0},
        {"burst_floor": 1.5},
    ],
)
def test_constructor_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        EQChannelSimulator(**kwargs)


def test_snr_helper() -> None:
    ref = np.array([1.0, -1.0, 1.0, -1.0])
    assert EQChannelSimulator.snr_db(ref, ref) == float("inf")
    assert EQChannelSimulator.snr_db(ref, ref + 0.1) == pytest.approx(20.0)
    with pytest.raises(ValueError):
        EQChannelSimulator.snr_db(ref, ref[:2])
