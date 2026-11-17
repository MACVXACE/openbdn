"""OpenBDN digital twin: an interactive Streamlit dashboard.

Launch with either of::

    streamlit run openbdn/app.py
    openbdn-digital-twin

The simulation helpers in this module (:func:`simulate_channel`,
:func:`windowed_snr_db`, :func:`sync_trace`) have no Streamlit dependency
and can be imported and tested on their own; Streamlit is imported only
inside :func:`main`.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from openbdn.channel.eq_hbc import EQChannelSimulator
from openbdn.crypto.vvtpm import SyncResult, VectorValuedTPM, synchronize
from openbdn.pipeline import BDNPipeline

__all__ = [
    "ChannelSnapshot",
    "SyncTrace",
    "launch",
    "main",
    "simulate_channel",
    "sync_trace",
    "windowed_snr_db",
]

FloatArray = NDArray[np.float64]

_BLUE, _ORANGE, _AQUA = "#2a78d6", "#eb6834", "#1baf7a"


@dataclass(frozen=True)
class ChannelSnapshot:
    """One simulated transmission, ready for plotting.

    Attributes
    ----------
    time_s:
        Sample times in seconds.
    clean, noisy, filtered:
        Transmitted waveform, received waveform, and NLMS output.
    snr_in_db, snr_out_db:
        Whole-record SNR before and after the equaliser.
    ber_unfiltered, ber_filtered:
        Bit error rate before and after the equaliser.
    window_time_s, window_snr_in_db, window_snr_out_db:
        Short-time SNR before and after the equaliser, and the centre time
        of each window.
    """

    time_s: FloatArray
    clean: FloatArray
    noisy: FloatArray
    filtered: FloatArray
    snr_in_db: float
    snr_out_db: float
    ber_unfiltered: float
    ber_filtered: float
    window_time_s: FloatArray
    window_snr_in_db: FloatArray
    window_snr_out_db: FloatArray

    @property
    def snr_gain_db(self) -> float:
        """SNR improvement delivered by the equaliser."""
        return self.snr_out_db - self.snr_in_db


@dataclass(frozen=True)
class SyncTrace:
    """History of one TPM key exchange.

    Attributes
    ----------
    result:
        Final outcome.
    overlap:
        Fraction of identical weights after each step; ``overlap[0]`` is the
        value before the first step.
    """

    result: SyncResult
    overlap: FloatArray


def windowed_snr_db(
    reference: FloatArray, estimate: FloatArray, window: int
) -> FloatArray:
    """Return the SNR of ``estimate`` in consecutive windows of ``window`` samples.

    A trailing partial window is dropped. Windows with zero error are
    reported as ``inf``.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have the same shape")
    count = reference.size // window
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    ref = reference[: count * window].reshape(count, window)
    err = (estimate[: count * window] - reference[: count * window]).reshape(count, window)
    signal_power = np.sum(np.square(ref), axis=1)
    error_power = np.sum(np.square(err), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(error_power > 0, signal_power / error_power, np.inf)
        return 10.0 * np.log10(ratio)


def simulate_channel(
    snr_db: float = 0.0,
    line_code: str = "manchester",
    n_bits: int = 2000,
    num_taps: int = 33,
    step_size: float = 0.05,
    burst_floor: float = 0.3,
    burst_rate_hz: float = 2.0,
    emg_corner_low_hz: float = 60.0,
    emg_corner_high_hz: float = 120.0,
    window_bits: int = 50,
    seed: int = 0,
) -> ChannelSnapshot:
    """Simulate one EMG-corrupted transmission and its NLMS recovery.

    Parameters mirror :class:`EQChannelSimulator`; ``window_bits`` sets the
    length of the short-time SNR windows.
    """
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")
    if window_bits < 1:
        raise ValueError("window_bits must be >= 1")
    channel = EQChannelSimulator(
        line_code=line_code,  # type: ignore[arg-type]
        num_taps=num_taps,
        step_size=step_size,
        burst_floor=burst_floor,
        burst_rate_hz=burst_rate_hz,
        emg_corner_low_hz=emg_corner_low_hz,
        emg_corner_high_hz=emg_corner_high_hz,
        seed=seed,
    )
    bits = np.random.default_rng(seed).integers(0, 2, n_bits, dtype=np.uint8)
    clean = channel.modulate(bits)
    noisy = channel.add_muscle_noise(clean, snr_db)
    filtered = channel.apply_adaptive_filter(noisy)

    window = window_bits * channel.samples_per_symbol
    snr_in_w = windowed_snr_db(clean, noisy, window)
    snr_out_w = windowed_snr_db(clean, filtered, window)
    centres = (np.arange(snr_in_w.size) + 0.5) * window / channel.sample_rate_hz

    return ChannelSnapshot(
        time_s=np.arange(clean.size) / channel.sample_rate_hz,
        clean=clean,
        noisy=noisy,
        filtered=filtered,
        snr_in_db=channel.snr_db(clean, noisy),
        snr_out_db=channel.snr_db(clean, filtered),
        ber_unfiltered=float(np.mean(channel.demodulate(noisy) != bits)),
        ber_filtered=float(np.mean(channel.demodulate(filtered) != bits)),
        window_time_s=centres,
        window_snr_in_db=snr_in_w,
        window_snr_out_db=snr_out_w,
    )


def sync_trace(
    k: int = 3,
    n: int = 100,
    l: int = 3,  # noqa: E741
    m: int = 1,
    rule: str = "hebbian",
    seed: int = 0,
    max_steps: int = 100_000,
    on_step: Callable[[int, float], None] | None = None,
) -> SyncTrace:
    """Run a TPM key exchange and record weight overlap after every step.

    ``on_step(step, overlap)`` is forwarded to :func:`synchronize` so a
    caller can refresh a live display while the exchange runs.
    """
    streams = np.random.SeedSequence(seed).spawn(3)
    alice = VectorValuedTPM(k, n, l, m, rule=rule, seed=np.random.default_rng(streams[0]))  # type: ignore[arg-type]
    bob = VectorValuedTPM(k, n, l, m, rule=rule, seed=np.random.default_rng(streams[1]))  # type: ignore[arg-type]
    history = [alice.overlap(bob)]

    def record(step: int, overlap: float) -> None:
        history.append(overlap)
        if on_step is not None:
            on_step(step, overlap)

    result = synchronize(
        alice, bob, max_steps=max_steps, rng=np.random.default_rng(streams[2]), on_step=record
    )
    return SyncTrace(result=result, overlap=np.asarray(history, dtype=np.float64))


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def _channel_tab(st, pd) -> None:  # type: ignore[no-untyped-def]
    """Render the EQ-HBC channel tab."""
    controls, view = st.columns([1, 3], gap="large")
    with controls:
        st.subheader("Channel")
        snr_db = st.slider("Channel SNR (dB)", -15.0, 20.0, 0.0, 0.5, key="ch_snr")
        line_code = st.radio(
            "Line code", ["manchester", "nrz"], horizontal=True, key="ch_code"
        )
        st.subheader("EMG interference")
        burst_floor = st.slider(
            "Resting EMG level",
            0.0,
            1.0,
            0.3,
            0.05,
            key="ch_floor",
            help="Relative EMG amplitude between contractions. 1 = stationary noise.",
        )
        burst_rate = st.slider("Contraction rate (Hz)", 0.5, 10.0, 2.0, 0.5, key="ch_rate")
        corners = st.slider(
            "EMG spectral corners f_l, f_h (Hz)", 10, 500, (60, 120), 5, key="ch_corners"
        )
        st.subheader("NLMS equaliser")
        taps = st.slider("Filter taps", 3, 129, 33, 2, key="ch_taps")
        step_size = st.slider("Step size μ", 0.005, 1.0, 0.05, 0.005, key="ch_mu")
        n_bits = st.slider("Bits simulated", 200, 4000, 2000, 100, key="ch_bits")
        seed = st.number_input("Seed", 0, 10_000, 0, key="ch_seed")

    if corners[0] >= corners[1]:
        view.error("The EMG corner frequencies must differ (f_l < f_h).")
        return

    snap = simulate_channel(
        snr_db=snr_db,
        line_code=line_code,
        n_bits=int(n_bits),
        num_taps=int(taps),
        step_size=float(step_size),
        burst_floor=float(burst_floor),
        burst_rate_hz=float(burst_rate),
        emg_corner_low_hz=float(corners[0]),
        emg_corner_high_hz=float(corners[1]),
        seed=int(seed),
    )

    with view:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("SNR before NLMS", f"{round(snap.snr_in_db, 1) + 0.0:.1f} dB")
        m2.metric("SNR after NLMS", f"{round(snap.snr_out_db, 1) + 0.0:.1f} dB", f"{snap.snr_gain_db:+.1f} dB")
        m3.metric("BER before NLMS", f"{snap.ber_unfiltered:.4f}")
        m4.metric(
            "BER after NLMS",
            f"{snap.ber_filtered:.4f}",
            f"{snap.ber_filtered - snap.ber_unfiltered:+.4f}" if snap.ber_filtered != snap.ber_unfiltered else None,
            delta_color="inverse",
        )

        st.markdown("**Waveforms**")
        duration_ms = float(snap.time_s[-1] * 1000.0)
        span_ms = st.slider(
            "Window length (ms)", 5.0, min(200.0, duration_ms), min(40.0, duration_ms), 5.0,
            key="ch_span",
        )
        start_ms = st.slider(
            "Window start (ms)", 0.0, max(duration_ms - span_ms, 0.0), 0.0, 5.0, key="ch_start"
        )
        mask = (snap.time_s * 1000.0 >= start_ms) & (snap.time_s * 1000.0 <= start_ms + span_ms)
        waves = pd.DataFrame(
            {
                "Time (ms)": snap.time_s[mask] * 1000.0,
                "Transmitted": snap.clean[mask],
                "Received (raw)": snap.noisy[mask],
                "After NLMS": snap.filtered[mask],
            }
        )
        st.line_chart(
            waves,
            x="Time (ms)",
            y=["Transmitted", "Received (raw)", "After NLMS"],
            color=[_BLUE, _ORANGE, _AQUA],
            y_label="Amplitude",
        )

        st.markdown("**Short-time SNR** (50-bit windows)")
        finite_out = np.where(np.isfinite(snap.window_snr_out_db), snap.window_snr_out_db, np.nan)
        running = pd.DataFrame(
            {
                "Time (ms)": snap.window_time_s * 1000.0,
                "Before NLMS": snap.window_snr_in_db,
                "After NLMS": finite_out,
            }
        )
        st.line_chart(
            running,
            x="Time (ms)",
            y=["Before NLMS", "After NLMS"],
            color=[_ORANGE, _AQUA],
            y_label="SNR (dB)",
        )
        st.caption(
            "The equaliser starts as an identity filter, so the two curves begin together and "
            "separate as the taps converge. Dips in the 'before' curve are muscle contractions."
        )
        with st.expander("Data table"):
            st.dataframe(running, hide_index=True, width="stretch")


def _sync_tab(st, pd) -> None:  # type: ignore[no-untyped-def]
    """Render the TPM key-exchange tab."""
    controls, view = st.columns([1, 3], gap="large")
    with controls:
        st.subheader("Tree Parity Machines")
        k = st.slider("Hidden units K", 1, 9, 3, key="tpm_k")
        n = st.slider("Inputs per unit N", 4, 200, 100, 4, key="tpm_n")
        depth = st.slider("Synaptic depth L", 1, 8, 3, key="tpm_l")
        m = st.slider("Input bound M", 1, 4, 1, key="tpm_m")
        rule = st.selectbox(
            "Learning rule", ["hebbian", "anti_hebbian", "random_walk"], key="tpm_rule"
        )
        max_steps = st.select_slider(
            "Step budget", [1_000, 5_000, 20_000, 100_000], value=20_000, key="tpm_budget"
        )
        seed = st.number_input("Seed", 0, 10_000, 0, key="tpm_seed")
        run = st.button("Run key exchange", type="primary", key="tpm_run")

    with view:
        status = st.empty()
        chart = st.empty()

        def draw(overlap: list[float] | np.ndarray) -> None:
            values = np.asarray(overlap, dtype=np.float64)
            stride = max(1, values.size // 600)
            frame = pd.DataFrame(
                {
                    "Step": np.arange(values.size)[::stride],
                    "Weight overlap (%)": values[::stride] * 100.0,
                }
            )
            chart.line_chart(frame, x="Step", y="Weight overlap (%)", color=_BLUE)

        if run:
            live: list[float] = []
            refresh = max(10, int(max_steps) // 400)

            def on_step(step: int, overlap: float) -> None:
                live.append(overlap)
                if step % refresh == 0:
                    status.progress(min(overlap, 1.0), text=f"step {step:,}: {overlap:.1%} overlap")
                    draw(live)

            st.session_state["tpm_trace"] = sync_trace(
                int(k), int(n), int(depth), int(m), str(rule), int(seed), int(max_steps), on_step
            )
            status.empty()

        trace: SyncTrace | None = st.session_state.get("tpm_trace")
        if trace is None:
            status.info("Choose an architecture and press **Run key exchange**.")
            return

        draw(trace.overlap)
        outcome = trace.result
        c1, c2, c3 = st.columns(3)
        c1.metric("Steps exchanged", f"{outcome.steps:,}")
        c2.metric("Weight updates", f"{outcome.updates:,}")
        c3.metric("Final overlap", f"{outcome.overlap:.1%}")
        if outcome.synchronized:
            st.success("Alice and Bob hold identical weights.")
            st.code(outcome.key or "", language=None)
        else:
            st.warning(f"Not synchronised within {outcome.steps:,} steps.")
        st.caption(
            "Neural key exchange is a research construction with published attacks; "
            "it must not be used to protect real data."
        )


def _pipeline_tab(st) -> None:  # type: ignore[no-untyped-def]
    """Render the end-to-end transmission tab."""
    message = st.text_input("Message", "Hello, BDN-04", key="e2e_msg")
    c1, c2, c3 = st.columns(3)
    snr_db = c1.slider("Channel SNR (dB)", -25.0, 20.0, 0.0, 0.5, key="e2e_snr")
    line_code = c2.radio("Line code", ["manchester", "nrz"], horizontal=True, key="e2e_code")
    seed = c3.number_input("Seed", 0, 10_000, 7, key="e2e_seed")
    if not message:
        st.info("Enter a message to transmit.")
        return

    channel = EQChannelSimulator(line_code=line_code, seed=int(seed))
    report = BDNPipeline(channel=channel, seed=int(seed)).transmit(message, snr_db=snr_db)

    st.markdown("**DNA strand**")
    st.code(report.dna, language=None, wrap_lines=True)
    a, b, c, d = st.columns(4)
    a.metric("Length", f"{len(report.dna)} nt")
    b.metric("GC-content", f"{report.gc_content:.1%}")
    c.metric("Longest homopolymer", report.max_homopolymer)
    d.metric("SNR gain", f"{report.snr_gain_db:+.1f} dB")
    if report.success:
        st.success(f"Recovered: {report.recovered_text}")
    else:
        detail = report.error or "decoded text differs from the original"
        st.error(f"Transmission failed ({detail}). The link has no error correction.")


def main() -> None:
    """Build the dashboard. Called when the file is run by Streamlit."""
    import pandas as pd
    import streamlit as st

    st.set_page_config(page_title="OpenBDN digital twin", layout="wide")
    st.title("OpenBDN digital twin")
    st.caption(
        "A numerical model of the BDN-04 data path: EQ-HBC channel with EMG interference, "
        "Tree Parity Machine key agreement, and constrained DNA storage coding."
    )
    channel_tab, sync_tab, pipeline_tab = st.tabs(
        ["EQ-HBC channel", "TPM key exchange", "End-to-end transmission"]
    )
    with channel_tab:
        _channel_tab(st, pd)
    with sync_tab:
        _sync_tab(st, pd)
    with pipeline_tab:
        _pipeline_tab(st)


def launch() -> None:
    """Console entry point ``openbdn-digital-twin``: start the Streamlit server."""
    try:
        from streamlit.web import cli as stcli
    except ImportError:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "Streamlit is not installed. Install it with: pip install 'openbdn[app]'"
        ) from None
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(stcli.main())


if __name__ == "__main__":
    main()
