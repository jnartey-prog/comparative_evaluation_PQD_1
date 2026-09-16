"""Scientific transform strategy implementations."""

from __future__ import annotations

import numpy as np
import pywt
from scipy import signal as scipy_signal
from vmdpy import VMD

from .models import Representation, SignalRecord


def transform(
    signal: SignalRecord, method: str, config: dict[str, float | str] | None = None
) -> Representation:
    """Apply STFT, CWT, selected-frequency S-transform, or VMD."""
    cfg = dict(config or {})
    method = method.lower()
    frequency_min = float(cfg.get("frequency_min", 0.0))
    frequency_max = min(float(cfg.get("frequency_max", 3000.0)), signal.sample_rate / 2)
    size = int(cfg.get("window", min(512, max(32, round(signal.sample_rate / 50.0)))))
    hop = int(cfg.get("hop", max(1, size // 4)))
    if method == "stft":
        window_name = str(cfg.get("window_name", "hann"))
        window: str | tuple[str, float] = (
            ("gaussian", float(cfg.get("window_std", max(1.0, size / 6))))
            if window_name == "gaussian"
            else window_name
        )
        freqs, times, coefficients = scipy_signal.stft(
            signal.samples,
            fs=signal.sample_rate,
            window=window,
            nperseg=size,
            noverlap=size - hop,
            nfft=int(2 ** np.ceil(np.log2(size))),
            boundary="even",
        )
        values = np.abs(coefficients)
        keep = (freqs >= frequency_min) & (freqs <= frequency_max)
        freqs, values = freqs[keep], values[keep]
    elif method == "wavelet":
        frequency_bins = int(cfg.get("frequency_bins", 128))
        minimum = max(float(cfg.get("frequency_min", 10.0)), 1.0 / signal.time[-1])
        target_frequencies = np.geomspace(minimum, frequency_max, frequency_bins)
        wavelet = str(cfg.get("wavelet", "cmor1.5-1.0"))
        center = pywt.central_frequency(wavelet)
        scales = center * signal.sample_rate / target_frequencies
        coefficients, freqs = pywt.cwt(
            signal.samples,
            scales,
            wavelet,
            sampling_period=1.0 / signal.sample_rate,
            method="fft",
        )
        values = np.abs(coefficients)
        times = signal.time
    elif method == "s_transform":
        values, freqs = _stockwell_selected(
            signal.samples,
            signal.sample_rate,
            int(cfg.get("frequency_bins", 128)),
            frequency_min,
            frequency_max,
            float(cfg.get("gaussian_width_factor", 1.0)),
        )
        times = signal.time
    elif method == "vmd":
        modes = int(cfg.get("modes", 5))
        values, _, omega = VMD(
            signal.samples,
            float(cfg.get("alpha", 2000.0)),
            float(cfg.get("tau", 0.0)),
            modes,
            int(cfg.get("dc", 0)),
            int(cfg.get("init", 1)),
            float(cfg.get("tolerance", 1e-7)),
        )
        freqs = omega[-1] * signal.sample_rate
        times = signal.time[: values.shape[1]]
        keep = (freqs >= frequency_min) & (freqs <= frequency_max)
        freqs, values = freqs[keep], values[keep]
    else:
        raise ValueError("method must be stft, wavelet, s_transform, or vmd")
    metadata = {"window": float(size), "hop": float(hop)}
    metadata.update({key: float(value) for key, value in cfg.items() if not isinstance(value, str)})
    return Representation(method, values, times, freqs, metadata)


def _stockwell_selected(
    samples: np.ndarray,
    sample_rate: float,
    requested_bins: int,
    frequency_min: float = 0.0,
    frequency_max: float | None = None,
    gaussian_width_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an explicit selected-frequency discrete Stockwell transform."""
    n_samples = len(samples)
    spectrum = np.fft.fft(samples)
    maximum = sample_rate / 2 if frequency_max is None else min(frequency_max, sample_rate / 2)
    positive = np.arange(1, n_samples // 2 + 1)
    frequencies_all = positive * sample_rate / n_samples
    positive = positive[(frequencies_all >= frequency_min) & (frequencies_all <= maximum)]
    if not len(positive):
        raise ValueError("requested S-transform frequency support contains no DFT bins")
    selection = positive[
        np.unique(np.linspace(0, len(positive) - 1, min(requested_bins, len(positive)), dtype=int))
    ]
    gaussian_index = np.fft.fftfreq(n_samples) * n_samples
    output = np.empty((len(selection), n_samples), dtype=complex)
    for row, frequency_index in enumerate(selection):
        gaussian = np.exp(
            -2 * (np.pi**2) * (gaussian_index**2) / ((gaussian_width_factor * frequency_index) ** 2)
        )
        shifted = np.roll(spectrum, -frequency_index)
        output[row] = np.fft.ifft(shifted * gaussian)
    frequencies = selection * sample_rate / n_samples
    return np.abs(output), frequencies
