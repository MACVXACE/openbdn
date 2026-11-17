"""Baseband model of an electro-quasistatic human body communication channel.

In the electro-quasistatic (EQS) regime the body behaves as a lumped,
approximately flat-band conductor, so the dominant impairment modelled here is
not multipath but *additive biological interference*: electromyographic (EMG)
activity picked up by the same electrodes that carry the data signal.

The simulator provides

* a non-stationary coloured-noise EMG generator
  (:meth:`EQChannelSimulator.add_muscle_noise`), and
* a Normalised Least Mean Squares (NLMS) adaptive equaliser
  (:meth:`EQChannelSimulator.apply_adaptive_filter`) that recovers the
  transmitted two-level waveform.

All quantities are real-valued baseband samples.
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal as sp_signal

FloatArray = NDArray[np.float64]
BitArray = NDArray[np.uint8]
LineCode = Literal["manchester", "nrz"]

__all__ = ["EQChannelSimulator"]


class EQChannelSimulator:
    """Simulate a two-level baseband EQ-HBC link corrupted by muscle noise.

    Parameters
    ----------
    sample_rate_hz:
        Sampling rate of the simulated waveform.
    samples_per_symbol:
        Number of samples per transmitted bit. Must be even for Manchester
        coding.
    line_code:
        ``"manchester"`` (default) sends each bit as a high-low or low-high
        pair of half-bit chips. Its spectrum has a null at DC and peaks near
        the bit rate, which keeps the data out of the EMG band; this mirrors
        the DC-free signalling used on practical body-channel links.
        ``"nrz"`` holds one level per bit and overlaps the EMG band heavily.
    num_taps:
        Length of the NLMS transversal filter. The filter is centred, i.e. it
        observes ``num_taps // 2`` samples either side of the sample being
        estimated, which corresponds to a decision latency of
        ``num_taps // 2`` samples in a real-time implementation.
    step_size:
        NLMS step size ``mu``. Stability requires ``0 < mu < 2``.
    regularization:
        Small positive constant ``epsilon`` that keeps the normalisation
        finite when the regressor energy is close to zero.
    levels:
        The two transmit amplitudes ``(low, high)`` assigned to bits 0 and 1.
    emg_corner_low_hz, emg_corner_high_hz:
        Corner frequencies ``f_l`` and ``f_h`` of the EMG spectral shaping
        function (see :meth:`emg_power_spectrum`).
    burst_rate_hz:
        Mean rate at which the simulated muscle switches between rest and
        contraction.
    burst_floor:
        Relative EMG amplitude at rest, in ``[0, 1]``. ``1`` gives stationary
        noise; smaller values give more pronounced movement bursts.
    seed:
        Seed or :class:`numpy.random.Generator` for reproducible noise.
    """

    def __init__(
        self,
        sample_rate_hz: float = 8000.0,
        samples_per_symbol: int = 4,
        line_code: LineCode = "manchester",
        num_taps: int = 33,
        step_size: float = 0.05,
        regularization: float = 1e-6,
        levels: tuple[float, float] = (-1.0, 1.0),
        emg_corner_low_hz: float = 60.0,
        emg_corner_high_hz: float = 120.0,
        burst_rate_hz: float = 2.0,
        burst_floor: float = 0.3,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if samples_per_symbol < 1:
            raise ValueError("samples_per_symbol must be >= 1")
        if line_code not in ("manchester", "nrz"):
            raise ValueError("line_code must be 'manchester' or 'nrz'")
        if line_code == "manchester" and samples_per_symbol % 2:
            raise ValueError("Manchester coding needs an even samples_per_symbol")
        if num_taps < 1:
            raise ValueError("num_taps must be >= 1")
        if not 0.0 < step_size < 2.0:
            raise ValueError("step_size must satisfy 0 < mu < 2 for NLMS stability")
        if regularization <= 0:
            raise ValueError("regularization must be positive")
        if levels[0] >= levels[1]:
            raise ValueError("levels must be given as (low, high) with low < high")
        if not 0 < emg_corner_low_hz < emg_corner_high_hz < sample_rate_hz / 2:
            raise ValueError("EMG corners must satisfy 0 < f_l < f_h < Nyquist")
        if burst_rate_hz <= 0:
            raise ValueError("burst_rate_hz must be positive")
        if not 0.0 <= burst_floor <= 1.0:
            raise ValueError("burst_floor must lie in [0, 1]")

        self.sample_rate_hz = float(sample_rate_hz)
        self.samples_per_symbol = int(samples_per_symbol)
        self.line_code: LineCode = line_code
        self.num_taps = int(num_taps)
        self.step_size = float(step_size)
        self.regularization = float(regularization)
        self.levels = (float(levels[0]), float(levels[1]))
        self.emg_corner_low_hz = float(emg_corner_low_hz)
        self.emg_corner_high_hz = float(emg_corner_high_hz)
        self.burst_rate_hz = float(burst_rate_hz)
        self.burst_floor = float(burst_floor)
        self._rng = np.random.default_rng(seed)
        #: Tap weights after the most recent call to
        #: :meth:`apply_adaptive_filter` (``None`` before the first call).
        self.taps: FloatArray | None = None

    # ------------------------------------------------------------------
    # Line coding
    # ------------------------------------------------------------------
    def modulate(self, bits: ArrayLike) -> FloatArray:
        """Map bits to an oversampled two-level waveform.

        With Manchester coding a 1 is sent as high-then-low and a 0 as
        low-then-high, each half lasting ``samples_per_symbol // 2`` samples.
        With NRZ the level is held for the whole bit.

        Parameters
        ----------
        bits:
            Sequence of 0/1 values.

        Returns
        -------
        numpy.ndarray
            Waveform of length ``len(bits) * samples_per_symbol``.
        """
        b = np.asarray(bits)
        if b.ndim != 1:
            raise ValueError("bits must be one-dimensional")
        if b.size and not np.isin(b, (0, 1)).all():
            raise ValueError("bits must contain only 0 and 1")
        low, high = self.levels
        ones = b.astype(bool)
        if self.line_code == "nrz":
            symbols = np.where(ones, high, low)
            return np.repeat(symbols, self.samples_per_symbol).astype(np.float64)
        chips = np.empty((b.size, 2), dtype=np.float64)
        chips[:, 0] = np.where(ones, high, low)
        chips[:, 1] = np.where(ones, low, high)
        return np.repeat(chips.ravel(), self.samples_per_symbol // 2)

    def demodulate(self, waveform: ArrayLike) -> BitArray:
        """Recover bits with a matched (integrate-and-dump) detector.

        NRZ: the mean of each bit period is compared with the mid-point of
        the two transmit levels. Manchester: the mean of the first half-bit
        is compared with the mean of the second, which needs no threshold
        and rejects slowly varying interference.
        """
        x = self._as_signal(waveform, allow_empty=True)
        if x.size % self.samples_per_symbol:
            raise ValueError("waveform length must be a multiple of samples_per_symbol")
        if x.size == 0:
            return np.zeros(0, dtype=np.uint8)
        frames = x.reshape(-1, self.samples_per_symbol)
        if self.line_code == "nrz":
            return (frames.mean(axis=1) > self._threshold).astype(np.uint8)
        half = self.samples_per_symbol // 2
        first = frames[:, :half].mean(axis=1)
        second = frames[:, half:].mean(axis=1)
        return (first > second).astype(np.uint8)

    # ------------------------------------------------------------------
    # Noise model
    # ------------------------------------------------------------------
    def emg_power_spectrum(self, freqs_hz: ArrayLike) -> FloatArray:
        r"""Evaluate the (unnormalised) surface-EMG power spectral density.

        The rational shaping function widely used for surface EMG is

        .. math::

            P(f) = \frac{f_h^4 \, f^2}{(f^2 + f_l^2)\,(f^2 + f_h^2)^2},

        which is zero at DC, rises to a peak between ``f_l`` and ``f_h`` and
        rolls off at 12 dB/octave above ``f_h``.
        """
        f2 = np.square(np.asarray(freqs_hz, dtype=np.float64))
        fl2 = self.emg_corner_low_hz**2
        fh2 = self.emg_corner_high_hz**2
        return (fh2**2) * f2 / ((f2 + fl2) * np.square(f2 + fh2))

    def add_muscle_noise(self, signal: ArrayLike, snr_db: float) -> FloatArray:
        r"""Add synthetic EMG interference at a prescribed signal-to-noise ratio.

        The artefact is generated in three steps:

        1. *Spectral colouring.* White Gaussian noise is shaped in the
           frequency domain by :math:`\sqrt{P(f)}`, with :math:`P(f)` from
           :meth:`emg_power_spectrum`, giving a stationary process with an
           EMG-like spectrum.
        2. *Movement modulation.* The coloured noise is multiplied by a
           smoothed random telegraph envelope that switches between rest
           (``burst_floor``) and contraction (1), making the interference
           non-stationary as it is during voluntary movement.
        3. *Scaling.* The artefact is scaled so that
           :math:`10\log_{10}(P_\text{signal} / P_\text{noise})` equals
           ``snr_db`` exactly over the supplied record.

        Parameters
        ----------
        signal:
            Clean transmitted waveform (one-dimensional).
        snr_db:
            Target SNR in decibels, measured over the whole record.

        Returns
        -------
        numpy.ndarray
            ``signal`` plus the EMG artefact; same shape as ``signal``.

        Raises
        ------
        ValueError
            If the signal is empty, not one-dimensional, non-finite, or has
            zero power (for which SNR is undefined).
        """
        x = self._as_signal(signal)
        if not np.isfinite(snr_db):
            raise ValueError("snr_db must be finite")
        signal_power = float(np.mean(np.square(x)))
        if signal_power == 0.0:
            raise ValueError("signal has zero power; SNR is undefined")

        noise = self._coloured_noise(x.size) * self._burst_envelope(x.size)
        noise_power = float(np.mean(np.square(noise)))
        if noise_power == 0.0:  # degenerate one- or two-sample records
            noise = self._rng.standard_normal(x.size)
            noise_power = float(np.mean(np.square(noise)))

        target_power = signal_power / (10.0 ** (snr_db / 10.0))
        return x + noise * np.sqrt(target_power / noise_power)

    def _coloured_noise(self, n: int) -> FloatArray:
        """Return ``n`` samples of Gaussian noise with the EMG spectrum."""
        white = self._rng.standard_normal(n)
        spectrum = np.fft.rfft(white)
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate_hz)
        return np.fft.irfft(spectrum * np.sqrt(self.emg_power_spectrum(freqs)), n=n)

    def _burst_envelope(self, n: int) -> FloatArray:
        """Return a smooth rest/contraction amplitude envelope in [floor, 1]."""
        mean_dwell = max(1.0, self.sample_rate_hz / self.burst_rate_hz)
        gate = np.empty(n, dtype=np.float64)
        position = 0
        active = bool(self._rng.integers(0, 2))
        while position < n:
            dwell = int(self._rng.exponential(mean_dwell)) + 1
            gate[position : position + dwell] = 1.0 if active else 0.0
            position += dwell
            active = not active

        # One-pole low-pass (20 ms time constant) models the finite rise time
        # of motor-unit recruitment.
        alpha = float(np.exp(-1.0 / (0.02 * self.sample_rate_hz)))
        b, a = [1.0 - alpha], [1.0, -alpha]
        zi = sp_signal.lfilter_zi(b, a) * gate[0]
        smooth, _ = sp_signal.lfilter(b, a, gate, zi=zi)
        return self.burst_floor + (1.0 - self.burst_floor) * smooth

    # ------------------------------------------------------------------
    # Adaptive equalisation
    # ------------------------------------------------------------------
    def apply_adaptive_filter(
        self,
        noisy_signal: ArrayLike,
        training: ArrayLike | None = None,
    ) -> FloatArray:
        r"""Denoise a received waveform with an NLMS adaptive equaliser.

        For every sample index :math:`n` the filter forms the regressor
        :math:`\mathbf{u}_n = [x_{n-c}, \dots, x_n, \dots, x_{n+c}]^\top`
        (``c = num_taps // 2``, zero-padded at the record edges) and applies
        the NLMS recursion

        .. math::

            y_n &= \mathbf{w}_n^\top \mathbf{u}_n, \\
            e_n &= d_n - y_n, \\
            \mathbf{w}_{n+1} &= \mathbf{w}_n
                + \frac{\mu \, e_n \, \mathbf{u}_n}
                       {\varepsilon + \mathbf{u}_n^\top \mathbf{u}_n}.

        The desired response :math:`d_n` is the known ``training`` sample
        while one is available (a preamble) and otherwise the *decision* on
        :math:`y_n`, i.e. the nearer of the two transmit levels
        (decision-directed mode). Because the EMG artefact is strongly
        coloured while the data waveform is wideband, the taps converge
        towards the Wiener solution that suppresses the EMG band.

        The filter is initialised to the identity (centre tap 1), so the
        output starts as the unprocessed input and improves as it adapts.

        Parameters
        ----------
        noisy_signal:
            Received waveform.
        training:
            Optional clean reference for the first ``len(training)`` samples.

        Returns
        -------
        numpy.ndarray
            Equalised waveform, time-aligned with ``noisy_signal``.
        """
        x = self._as_signal(noisy_signal)
        if training is None:
            reference = np.zeros(0, dtype=np.float64)
        else:
            reference = self._as_signal(training, allow_empty=True)
            if reference.size > x.size:
                raise ValueError("training sequence is longer than the signal")

        taps = self.num_taps
        centre = taps // 2
        padded = np.concatenate(
            (np.zeros(centre), x, np.zeros(taps - 1 - centre))
        )
        weights = np.zeros(taps, dtype=np.float64)
        weights[centre] = 1.0

        low, high = self.levels
        threshold = self._threshold
        mu, eps = self.step_size, self.regularization
        n_train = reference.size
        output = np.empty_like(x)

        for n in range(x.size):
            u = padded[n : n + taps]
            y = float(weights @ u)
            if n < n_train:
                desired = reference[n]
            else:
                desired = high if y > threshold else low
            error = desired - y
            weights += (mu * error / (eps + float(u @ u))) * u
            output[n] = y

        self.taps = weights
        return output

    # ------------------------------------------------------------------
    # Metrics and helpers
    # ------------------------------------------------------------------
    @staticmethod
    def snr_db(reference: ArrayLike, estimate: ArrayLike) -> float:
        """Return the SNR of ``estimate`` relative to the clean ``reference``.

        Defined as ``10 log10(sum(reference**2) / sum((estimate - reference)**2))``.
        Returns ``inf`` when the two are identical.
        """
        ref = np.asarray(reference, dtype=np.float64)
        est = np.asarray(estimate, dtype=np.float64)
        if ref.shape != est.shape:
            raise ValueError("reference and estimate must have the same shape")
        if ref.size == 0:
            raise ValueError("cannot compute SNR of an empty signal")
        error_power = float(np.sum(np.square(est - ref)))
        signal_power = float(np.sum(np.square(ref)))
        if error_power == 0.0:
            return float("inf")
        if signal_power == 0.0:
            raise ValueError("reference has zero power; SNR is undefined")
        return 10.0 * float(np.log10(signal_power / error_power))

    @property
    def _threshold(self) -> float:
        return 0.5 * (self.levels[0] + self.levels[1])

    @staticmethod
    def _as_signal(values: ArrayLike | Sequence[float], allow_empty: bool = False) -> FloatArray:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim != 1:
            raise ValueError("signal must be one-dimensional")
        if x.size == 0 and not allow_empty:
            raise ValueError("signal must not be empty")
        if not np.isfinite(x).all():
            raise ValueError("signal contains NaN or infinite values")
        return x
