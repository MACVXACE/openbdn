"""Automated benchmarking of the OpenBDN channel and key-agreement layers.

Two experiments are provided:

* :func:`benchmark_channel` sweeps the input SNR and compares both line
  codes on bit error rate (BER) and on the SNR improvement delivered by the
  NLMS equaliser.
* :func:`benchmark_sync` measures how many public inputs two Tree Parity
  Machines need to synchronise, across hidden-layer sizes and learning rules.

:func:`run_benchmark` runs both, writes one CSV per experiment and renders
summary plots with matplotlib.

Example
-------
>>> from openbdn.benchmark import BenchmarkConfig, run_benchmark
>>> report = run_benchmark("results", BenchmarkConfig.quick())  # doctest: +SKIP
>>> sorted(report.files)  # doctest: +SKIP
['channel_csv', 'channel_plot', 'sync_csv', 'sync_plot']
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from openbdn.channel.eq_hbc import EQChannelSimulator
from openbdn.crypto.vvtpm import VectorValuedTPM, synchronize

__all__ = [
    "LINE_CODES",
    "LEARNING_RULES",
    "BenchmarkConfig",
    "BenchmarkReport",
    "ChannelBenchmarkResult",
    "SyncBenchmarkResult",
    "benchmark_channel",
    "benchmark_sync",
    "plot_channel",
    "plot_sync",
    "run_benchmark",
    "snr_grid",
    "write_csv",
]

LINE_CODES: tuple[str, ...] = ("manchester", "nrz")
LEARNING_RULES: tuple[str, ...] = ("hebbian", "anti_hebbian", "random_walk")

ProgressCallback = Callable[[str, int, int], None]

# Categorical colours (fixed order) and chrome for the summary plots.
_SERIES_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a")
_SURFACE = "#fcfcfb"
_TEXT = "#0b0b0b"
_TEXT_MUTED = "#52514e"
_GRID = "#e4e3df"


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelBenchmarkResult:
    """Channel metrics for one ``(line_code, snr_db)`` operating point.

    Attributes
    ----------
    line_code:
        ``"manchester"`` or ``"nrz"``.
    snr_db:
        Requested channel SNR.
    trials:
        Number of independent noise realisations averaged.
    bits_per_trial:
        Payload length of each realisation.
    snr_in_db, snr_out_db:
        Mean waveform SNR before and after the NLMS equaliser.
    snr_gain_db:
        Mean per-trial improvement ``snr_out_db - snr_in_db``.
    ber_unfiltered, ber_filtered:
        Bit error rate, pooled over all trials, when detecting the raw and
        the equalised waveform.
    bit_errors_unfiltered, bit_errors_filtered:
        The underlying error counts.
    """

    line_code: str
    snr_db: float
    trials: int
    bits_per_trial: int
    snr_in_db: float
    snr_out_db: float
    snr_gain_db: float
    ber_unfiltered: float
    ber_filtered: float
    bit_errors_unfiltered: int
    bit_errors_filtered: int


@dataclass(frozen=True)
class SyncBenchmarkResult:
    """Synchronisation statistics for one TPM configuration.

    Step statistics are computed over the trials that synchronised within
    ``max_steps``; they are ``nan`` when none did.

    Attributes
    ----------
    k, n, l, m:
        Architecture: hidden units, inputs per unit, synaptic depth, input
        magnitude bound.
    rule:
        Learning rule.
    trials:
        Number of independent key exchanges attempted.
    max_steps:
        Step budget per exchange.
    synchronized:
        Number of exchanges that reached identical weights.
    success_rate:
        ``synchronized / trials``.
    steps_mean, steps_median, steps_std, steps_min, steps_max:
        Statistics of the number of exchanged inputs.
    """

    k: int
    n: int
    l: int  # noqa: E741
    m: int
    rule: str
    trials: int
    max_steps: int
    synchronized: int
    success_rate: float
    steps_mean: float
    steps_median: float
    steps_std: float
    steps_min: float
    steps_max: float


@dataclass(frozen=True)
class BenchmarkConfig:
    """Parameters of a full benchmark run.

    Attributes
    ----------
    snrs_db:
        Channel SNRs to sweep. Defaults to -5 ... +15 dB in 2.5 dB steps.
    line_codes:
        Line codes to compare.
    channel_trials:
        Noise realisations per channel operating point.
    bits_per_trial:
        Payload bits per realisation.
    hidden_units:
        Values of ``K`` for the TPM experiment.
    rules:
        Learning rules for the TPM experiment.
    tpm_n, tpm_l, tpm_m:
        Remaining TPM architecture parameters.
    sync_trials:
        Key exchanges per TPM configuration.
    max_steps:
        Step budget per key exchange.
    seed:
        Master seed; every trial derives its own stream from it.
    """

    snrs_db: tuple[float, ...] = field(default_factory=lambda: snr_grid())
    line_codes: tuple[str, ...] = LINE_CODES
    channel_trials: int = 8
    bits_per_trial: int = 4000
    hidden_units: tuple[int, ...] = (3, 5, 7)
    rules: tuple[str, ...] = LEARNING_RULES
    tpm_n: int = 100
    tpm_l: int = 3
    tpm_m: int = 1
    sync_trials: int = 10
    max_steps: int = 200_000
    seed: int = 0

    @classmethod
    def quick(cls, seed: int = 0) -> "BenchmarkConfig":
        """Return a reduced configuration that finishes in a few seconds."""
        return cls(
            snrs_db=snr_grid(step=5.0),
            channel_trials=2,
            bits_per_trial=1000,
            tpm_n=16,
            sync_trials=3,
            max_steps=50_000,
            seed=seed,
        )


@dataclass(frozen=True)
class BenchmarkReport:
    """Output of :func:`run_benchmark`.

    Attributes
    ----------
    channel:
        One row per ``(line_code, snr_db)``.
    sync:
        One row per ``(k, rule)``.
    files:
        Paths of the artefacts written, keyed by ``channel_csv``,
        ``sync_csv`` and, when plots were requested, ``channel_plot`` and
        ``sync_plot``.
    """

    channel: list[ChannelBenchmarkResult]
    sync: list[SyncBenchmarkResult]
    files: dict[str, Path]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
def snr_grid(start: float = -5.0, stop: float = 15.0, step: float = 2.5) -> tuple[float, ...]:
    """Return an inclusive, evenly spaced SNR grid in decibels.

    Raises
    ------
    ValueError
        If ``step`` is not positive, ``stop < start``, or any value is not
        finite.
    """
    if not all(math.isfinite(v) for v in (start, stop, step)):
        raise ValueError("start, stop and step must be finite")
    if step <= 0:
        raise ValueError("step must be positive")
    if stop < start:
        raise ValueError("stop must be >= start")
    count = int(math.floor((stop - start) / step + 1e-9)) + 1
    return tuple(float(start + i * step) for i in range(count))


def benchmark_channel(
    snrs_db: Sequence[float] | None = None,
    line_codes: Sequence[str] = LINE_CODES,
    trials: int = 8,
    bits_per_trial: int = 4000,
    seed: int = 0,
    progress: ProgressCallback | None = None,
    **channel_kwargs: Any,
) -> list[ChannelBenchmarkResult]:
    """Sweep SNR and line code, measuring BER and equaliser SNR gain.

    Every operating point uses the same payload bits and the same noise
    seeds (common random numbers), so differences between line codes and
    SNRs are not blurred by sampling noise. The equaliser runs in
    decision-directed mode with no training preamble.

    Parameters
    ----------
    snrs_db:
        SNRs to evaluate; defaults to :func:`snr_grid`.
    line_codes:
        Subset of :data:`LINE_CODES`.
    trials:
        Independent noise realisations per operating point.
    bits_per_trial:
        Payload length per realisation.
    seed:
        Master seed.
    progress:
        Optional ``progress(stage, done, total)`` callback.
    **channel_kwargs:
        Extra keyword arguments for :class:`EQChannelSimulator`
        (e.g. ``num_taps`` or ``step_size``).

    Returns
    -------
    list[ChannelBenchmarkResult]
        Ordered by line code, then SNR as supplied.
    """
    snrs = snr_grid() if snrs_db is None else tuple(float(s) for s in snrs_db)
    codes = tuple(line_codes)
    if not snrs:
        raise ValueError("snrs_db must not be empty")
    if not all(math.isfinite(s) for s in snrs):
        raise ValueError("snrs_db must contain only finite values")
    if not codes:
        raise ValueError("line_codes must not be empty")
    unknown = sorted(set(codes) - set(LINE_CODES))
    if unknown:
        raise ValueError(f"unknown line code(s): {unknown}")
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if bits_per_trial < 1:
        raise ValueError("bits_per_trial must be >= 1")
    for reserved in ("line_code", "seed"):
        if reserved in channel_kwargs:
            raise ValueError(f"{reserved!r} is controlled by the benchmark")

    payloads = [
        np.random.default_rng([seed, trial, 0]).integers(0, 2, bits_per_trial, dtype=np.uint8)
        for trial in range(trials)
    ]
    total = len(codes) * len(snrs)
    done = 0
    results: list[ChannelBenchmarkResult] = []

    for code in codes:
        for snr in snrs:
            snr_in: list[float] = []
            snr_out: list[float] = []
            errors_raw = errors_filtered = 0
            for trial, bits in enumerate(payloads):
                channel = EQChannelSimulator(
                    line_code=code,  # type: ignore[arg-type]
                    seed=np.random.default_rng([seed, trial, 1]),
                    **channel_kwargs,
                )
                clean = channel.modulate(bits)
                noisy = channel.add_muscle_noise(clean, snr)
                filtered = channel.apply_adaptive_filter(noisy)
                snr_in.append(channel.snr_db(clean, noisy))
                snr_out.append(channel.snr_db(clean, filtered))
                errors_raw += int(np.sum(channel.demodulate(noisy) != bits))
                errors_filtered += int(np.sum(channel.demodulate(filtered) != bits))

            total_bits = trials * bits_per_trial
            results.append(
                ChannelBenchmarkResult(
                    line_code=code,
                    snr_db=snr,
                    trials=trials,
                    bits_per_trial=bits_per_trial,
                    snr_in_db=float(np.mean(snr_in)),
                    snr_out_db=float(np.mean(snr_out)),
                    snr_gain_db=float(np.mean(np.subtract(snr_out, snr_in))),
                    ber_unfiltered=errors_raw / total_bits,
                    ber_filtered=errors_filtered / total_bits,
                    bit_errors_unfiltered=errors_raw,
                    bit_errors_filtered=errors_filtered,
                )
            )
            done += 1
            if progress is not None:
                progress("channel", done, total)
    return results


def benchmark_sync(
    hidden_units: Sequence[int] = (3, 5, 7),
    rules: Sequence[str] = LEARNING_RULES,
    n: int = 100,
    l: int = 3,  # noqa: E741
    m: int = 1,
    trials: int = 10,
    max_steps: int = 200_000,
    seed: int = 0,
    progress: ProgressCallback | None = None,
) -> list[SyncBenchmarkResult]:
    """Measure TPM synchronisation time across ``K`` and learning rules.

    Parameters
    ----------
    hidden_units:
        Values of ``K`` to evaluate.
    rules:
        Subset of :data:`LEARNING_RULES`.
    n, l, m:
        Inputs per hidden unit, synaptic depth and input magnitude bound.
    trials:
        Independent key exchanges per configuration.
    max_steps:
        Step budget per exchange. Exchanges that exhaust it count as
        failures and are excluded from the step statistics.
    seed:
        Master seed.
    progress:
        Optional ``progress(stage, done, total)`` callback.

    Returns
    -------
    list[SyncBenchmarkResult]
        Ordered by ``K``, then rule as supplied.
    """
    ks = tuple(int(k) for k in hidden_units)
    rule_names = tuple(rules)
    if not ks:
        raise ValueError("hidden_units must not be empty")
    if min(ks) < 1:
        raise ValueError("hidden_units must be positive")
    if not rule_names:
        raise ValueError("rules must not be empty")
    unknown = sorted(set(rule_names) - set(LEARNING_RULES))
    if unknown:
        raise ValueError(f"unknown learning rule(s): {unknown}")
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    total = len(ks) * len(rule_names)
    done = 0
    results: list[SyncBenchmarkResult] = []

    for k in ks:
        for rule in rule_names:
            steps: list[int] = []
            for trial in range(trials):
                # Same initial weights and public inputs for every rule.
                alice = VectorValuedTPM(k, n, l, m, rule=rule, seed=np.random.default_rng([seed, k, trial, 0]))  # type: ignore[arg-type]
                bob = VectorValuedTPM(k, n, l, m, rule=rule, seed=np.random.default_rng([seed, k, trial, 1]))  # type: ignore[arg-type]
                outcome = synchronize(
                    alice, bob, max_steps=max_steps, rng=np.random.default_rng([seed, k, trial, 2])
                )
                if outcome.synchronized:
                    steps.append(outcome.steps)

            if steps:
                arr = np.asarray(steps, dtype=np.float64)
                stats = (
                    float(arr.mean()),
                    float(np.median(arr)),
                    float(arr.std()),
                    float(arr.min()),
                    float(arr.max()),
                )
            else:
                stats = (math.nan,) * 5
            results.append(
                SyncBenchmarkResult(
                    k=k,
                    n=n,
                    l=l,
                    m=m,
                    rule=rule,
                    trials=trials,
                    max_steps=max_steps,
                    synchronized=len(steps),
                    success_rate=len(steps) / trials,
                    steps_mean=stats[0],
                    steps_median=stats[1],
                    steps_std=stats[2],
                    steps_min=stats[3],
                    steps_max=stats[4],
                )
            )
            done += 1
            if progress is not None:
                progress("sync", done, total)
    return results


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.6g}"
    return str(value)


def write_csv(rows: Iterable[Any], path: str | Path) -> Path:
    """Write a homogeneous list of result dataclasses to a CSV file.

    The header is the dataclass field names. Floats are written with six
    significant digits and missing statistics as ``nan``, so every numeric
    column parses with :func:`float`. Parent directories are created.

    Raises
    ------
    ValueError
        If ``rows`` is empty or mixes result types.
    TypeError
        If a row is not a dataclass instance.
    """
    items = list(rows)
    if not items:
        raise ValueError("cannot write an empty result set")
    first = items[0]
    if not is_dataclass(first) or isinstance(first, type):
        raise TypeError("rows must be dataclass instances")
    if any(type(item) is not type(first) for item in items):
        raise ValueError("all rows must have the same type")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    names = [f.name for f in fields(first)]
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for item in items:
            writer.writerow({key: _format_cell(val) for key, val in asdict(item).items()})
    return target


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def _new_figure(width: float, height: float, ncols: int):  # type: ignore[no-untyped-def]
    """Create a headless matplotlib figure (no pyplot global state)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(width, height), dpi=150, facecolor=_SURFACE)
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, ncols, squeeze=False)[0]
    for ax in axes:
        ax.set_facecolor(_SURFACE)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)
        ax.tick_params(colors=_TEXT_MUTED, labelsize=9, length=0)
        ax.xaxis.label.set_color(_TEXT_MUTED)
        ax.yaxis.label.set_color(_TEXT_MUTED)
    return figure, axes


def _style_legend(legend) -> None:  # type: ignore[no-untyped-def]
    for text in legend.get_texts():
        text.set_color(_TEXT_MUTED)


def plot_channel(results: Sequence[ChannelBenchmarkResult], path: str | Path) -> Path:
    """Render the channel summary figure.

    Left panel: BER against input SNR on a logarithmic axis, one colour per
    line code, dashed before the NLMS filter and solid after it. Points
    with no observed errors are drawn on a labelled floor at half the
    measurement resolution. Right panel: equaliser SNR gain against input
    SNR.

    Raises
    ------
    ValueError
        If ``results`` is empty.
    """
    if not results:
        raise ValueError("no channel results to plot")
    codes = list(dict.fromkeys(r.line_code for r in results))
    colours = {
        code: _SERIES_COLOURS[LINE_CODES.index(code) % len(_SERIES_COLOURS)] for code in codes
    }
    floor = 0.5 / max(r.trials * r.bits_per_trial for r in results)

    figure, (ax_ber, ax_gain) = _new_figure(11.0, 4.4, 2)
    for code in codes:
        rows = sorted((r for r in results if r.line_code == code), key=lambda r: r.snr_db)
        snr = [r.snr_db for r in rows]
        colour = colours[code]
        ax_ber.plot(
            snr,
            [max(r.ber_unfiltered, floor) for r in rows],
            color=colour,
            linewidth=2,
            linestyle=(0, (4, 3)),
            marker="o",
            markersize=5,
            markerfacecolor=_SURFACE,
            label=f"{code}, before NLMS",
        )
        ax_ber.plot(
            snr,
            [max(r.ber_filtered, floor) for r in rows],
            color=colour,
            linewidth=2,
            marker="o",
            markersize=5,
            label=f"{code}, after NLMS",
        )
        gain = [r.snr_gain_db for r in rows]
        ax_gain.plot(snr, gain, color=colour, linewidth=2, marker="o", markersize=5, label=code)
        ax_gain.annotate(
            code,
            (snr[-1], gain[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=_TEXT_MUTED,
        )

    ax_ber.set_yscale("log")
    ax_ber.axhline(floor, color=_TEXT_MUTED, linewidth=0.8, linestyle=":")
    ax_ber.annotate(
        "no errors observed",
        (0.01, floor),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        fontsize=8,
        color=_TEXT_MUTED,
    )
    ax_ber.set_xlabel("Input SNR (dB)")
    ax_ber.set_ylabel("Bit error rate")
    ax_ber.set_title("Bit error rate", loc="left", fontsize=11, color=_TEXT)
    _style_legend(ax_ber.legend(frameon=False, fontsize=8, loc="upper right"))

    ax_gain.axhline(0.0, color=_TEXT_MUTED, linewidth=0.8)
    ax_gain.set_xlabel("Input SNR (dB)")
    ax_gain.set_ylabel("SNR gain (dB)")
    ax_gain.set_title("NLMS equaliser SNR gain", loc="left", fontsize=11, color=_TEXT)
    ax_gain.margins(x=0.12)
    _style_legend(ax_gain.legend(frameon=False, fontsize=8, loc="upper right"))

    figure.tight_layout()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, facecolor=_SURFACE)
    return target


def plot_sync(results: Sequence[SyncBenchmarkResult], path: str | Path) -> Path:
    """Render the TPM synchronisation figure.

    For every ``K`` the median number of steps is drawn as a marker per
    learning rule, with a vertical bar spanning the fastest and slowest
    trial, on a logarithmic axis. Configurations in which no trial
    synchronised are omitted from the plot and listed in a note under it.

    Raises
    ------
    ValueError
        If ``results`` is empty.
    """
    if not results:
        raise ValueError("no synchronisation results to plot")
    ks = sorted({r.k for r in results})
    rules = list(dict.fromkeys(r.rule for r in results))
    width = 0.6 / max(len(rules), 1)

    figure, (ax,) = _new_figure(7.5, 4.6, 1)
    failed: list[str] = []
    plotted = False
    for rule in rules:
        colour = _SERIES_COLOURS[LEARNING_RULES.index(rule) % len(_SERIES_COLOURS)]
        offset = (rules.index(rule) - (len(rules) - 1) / 2) * width
        xs: list[float] = []
        medians: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for row in (r for r in results if r.rule == rule):
            if row.synchronized == 0:
                failed.append(f"K={row.k} {rule}")
                continue
            xs.append(ks.index(row.k) + offset)
            medians.append(row.steps_median)
            lows.append(row.steps_median - row.steps_min)
            highs.append(row.steps_max - row.steps_median)
        if not xs:
            continue
        plotted = True
        ax.errorbar(
            xs,
            medians,
            yerr=[lows, highs],
            fmt="o",
            color=colour,
            markersize=8,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.5,
            elinewidth=2,
            capsize=0,
            label=rule.replace("_", "-"),
        )

    if plotted:
        ax.set_yscale("log")
        _style_legend(ax.legend(frameon=False, fontsize=9, loc="upper left"))
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([f"K = {k}" for k in ks])
    ax.set_xlim(-0.5, len(ks) - 0.5)
    ax.grid(False, axis="x")
    ax.set_ylabel("Steps to synchronise (median, min-max)")
    sample = results[0]
    ax.set_title(
        f"TPM synchronisation time (N={sample.n}, L={sample.l}, M={sample.m}, "
        f"{sample.trials} trials)",
        loc="left",
        fontsize=11,
        color=_TEXT,
    )
    if failed:
        figure.text(
            0.01,
            0.01,
            f"Not synchronised within {sample.max_steps:,} steps: " + ", ".join(failed),
            fontsize=8,
            color=_TEXT_MUTED,
        )

    figure.tight_layout(rect=(0, 0.04 if failed else 0, 1, 1))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, facecolor=_SURFACE)
    return target


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def run_benchmark(
    output_dir: str | Path = "benchmark_results",
    config: BenchmarkConfig | None = None,
    make_plots: bool = True,
    progress: ProgressCallback | None = None,
) -> BenchmarkReport:
    """Run both experiments and write CSVs and plots to ``output_dir``.

    Files written
    -------------
    ``channel_benchmark.csv``
        One row per line code and SNR.
    ``tpm_sync_benchmark.csv``
        One row per ``K`` and learning rule.
    ``channel_benchmark.png``, ``tpm_sync_benchmark.png``
        Summary figures (skipped when ``make_plots`` is false).

    Parameters
    ----------
    output_dir:
        Destination directory; created if missing.
    config:
        Run parameters; defaults to :class:`BenchmarkConfig`.
    make_plots:
        Whether to render the figures (requires matplotlib).
    progress:
        Optional ``progress(stage, done, total)`` callback.

    Returns
    -------
    BenchmarkReport
    """
    cfg = config if config is not None else BenchmarkConfig()
    out = Path(output_dir)
    if out.exists() and not out.is_dir():
        raise NotADirectoryError(f"{out} exists and is not a directory")
    out.mkdir(parents=True, exist_ok=True)

    channel = benchmark_channel(
        snrs_db=cfg.snrs_db,
        line_codes=cfg.line_codes,
        trials=cfg.channel_trials,
        bits_per_trial=cfg.bits_per_trial,
        seed=cfg.seed,
        progress=progress,
    )
    sync = benchmark_sync(
        hidden_units=cfg.hidden_units,
        rules=cfg.rules,
        n=cfg.tpm_n,
        l=cfg.tpm_l,
        m=cfg.tpm_m,
        trials=cfg.sync_trials,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
        progress=progress,
    )

    files = {
        "channel_csv": write_csv(channel, out / "channel_benchmark.csv"),
        "sync_csv": write_csv(sync, out / "tpm_sync_benchmark.csv"),
    }
    if make_plots:
        files["channel_plot"] = plot_channel(channel, out / "channel_benchmark.png")
        files["sync_plot"] = plot_sync(sync, out / "tpm_sync_benchmark.png")
    return BenchmarkReport(channel=channel, sync=sync, files=files)
