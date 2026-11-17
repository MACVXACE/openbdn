"""Tests for the benchmarking suite, its CLI command and the dashboard helpers."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from openbdn.benchmark import (
    LEARNING_RULES,
    LINE_CODES,
    BenchmarkConfig,
    ChannelBenchmarkResult,
    SyncBenchmarkResult,
    benchmark_channel,
    benchmark_sync,
    plot_channel,
    plot_sync,
    run_benchmark,
    snr_grid,
    write_csv,
)
from openbdn.cli import app

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def channel_results() -> list[ChannelBenchmarkResult]:
    return benchmark_channel(snrs_db=(-5.0, 5.0, 15.0), trials=2, bits_per_trial=1500, seed=1)


@pytest.fixture(scope="module")
def sync_results() -> list[SyncBenchmarkResult]:
    return benchmark_sync(hidden_units=(3, 5), n=16, trials=3, max_steps=50_000, seed=1)


# ------------------------------------------------------------------- SNR grid
def test_default_snr_grid_matches_specification() -> None:
    assert snr_grid() == (-5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0)


def test_snr_grid_variants() -> None:
    assert snr_grid(0.0, 0.0, 1.0) == (0.0,)
    assert snr_grid(0.0, 1.0, 0.4) == pytest.approx((0.0, 0.4, 0.8))
    assert snr_grid(-1.0, 1.0, 0.1)[-1] == pytest.approx(1.0)  # no float drop-out


@pytest.mark.parametrize(
    "args", [(0.0, 10.0, 0.0), (0.0, 10.0, -1.0), (5.0, 0.0, 1.0), (0.0, math.inf, 1.0)]
)
def test_snr_grid_rejects_bad_ranges(args: tuple[float, float, float]) -> None:
    with pytest.raises(ValueError):
        snr_grid(*args)


# ------------------------------------------------------------ channel sweep
def test_channel_benchmark_covers_every_operating_point(
    channel_results: list[ChannelBenchmarkResult],
) -> None:
    assert [(r.line_code, r.snr_db) for r in channel_results] == [
        (code, snr) for code in LINE_CODES for snr in (-5.0, 5.0, 15.0)
    ]
    for row in channel_results:
        assert row.trials == 2 and row.bits_per_trial == 1500
        assert 0.0 <= row.ber_filtered <= 1.0
        assert 0.0 <= row.ber_unfiltered <= 1.0
        assert row.ber_unfiltered == row.bit_errors_unfiltered / 3000
        assert row.ber_filtered == row.bit_errors_filtered / 3000
        assert row.snr_in_db == pytest.approx(row.snr_db, abs=1e-6)
        assert row.snr_gain_db == pytest.approx(row.snr_out_db - row.snr_in_db)


def test_channel_benchmark_reproduces_known_physics(
    channel_results: list[ChannelBenchmarkResult],
) -> None:
    by_key = {(r.line_code, r.snr_db): r for r in channel_results}
    for row in channel_results:
        assert row.snr_gain_db > 0.0  # the equaliser always helps here
    for snr in (-5.0, 5.0, 15.0):
        # DC-free Manchester keeps data out of the EMG band.
        assert by_key["manchester", snr].snr_gain_db > by_key["nrz", snr].snr_gain_db
        assert by_key["manchester", snr].ber_unfiltered <= by_key["nrz", snr].ber_unfiltered
    nrz = [by_key["nrz", snr] for snr in (-5.0, 5.0, 15.0)]
    assert nrz[0].ber_unfiltered > nrz[1].ber_unfiltered > nrz[2].ber_unfiltered
    assert nrz[0].ber_filtered < nrz[0].ber_unfiltered


def test_channel_benchmark_is_deterministic() -> None:
    kwargs = dict(snrs_db=(0.0,), trials=2, bits_per_trial=300)
    assert benchmark_channel(seed=3, **kwargs) == benchmark_channel(seed=3, **kwargs)
    assert benchmark_channel(seed=3, **kwargs) != benchmark_channel(seed=4, **kwargs)


def test_channel_benchmark_minimal_and_custom_inputs() -> None:
    rows = benchmark_channel(snrs_db=[40.0], line_codes=["nrz"], trials=1, bits_per_trial=1)
    assert len(rows) == 1 and rows[0].line_code == "nrz"
    assert math.isfinite(rows[0].snr_out_db)

    rows = benchmark_channel(
        snrs_db=(0.0,), line_codes=("manchester",), trials=1, bits_per_trial=200, num_taps=9
    )
    assert len(rows) == 1


def test_channel_benchmark_reports_progress() -> None:
    calls: list[tuple[str, int, int]] = []
    benchmark_channel(
        snrs_db=(0.0, 5.0), trials=1, bits_per_trial=100, progress=lambda *a: calls.append(a)
    )
    assert calls == [("channel", i, 4) for i in range(1, 5)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"snrs_db": ()},
        {"snrs_db": (math.nan,)},
        {"line_codes": ()},
        {"line_codes": ("ami",)},
        {"trials": 0},
        {"bits_per_trial": 0},
        {"seed": 1, "line_code": "nrz"},
    ],
)
def test_channel_benchmark_rejects_bad_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        benchmark_channel(**kwargs)


# --------------------------------------------------------------- TPM sweep
def test_sync_benchmark_covers_every_configuration(
    sync_results: list[SyncBenchmarkResult],
) -> None:
    assert [(r.k, r.rule) for r in sync_results] == [
        (k, rule) for k in (3, 5) for rule in LEARNING_RULES
    ]
    for row in sync_results:
        assert row.synchronized == row.trials == 3
        assert row.success_rate == 1.0
        assert 1 <= row.steps_min <= row.steps_median <= row.steps_max <= row.max_steps
        assert row.steps_min <= row.steps_mean <= row.steps_max
        assert row.steps_std >= 0.0


def test_sync_benchmark_is_deterministic() -> None:
    kwargs = dict(hidden_units=(3,), rules=("hebbian",), n=12, trials=2)
    assert benchmark_sync(seed=5, **kwargs) == benchmark_sync(seed=5, **kwargs)


def test_sync_benchmark_handles_exhausted_budget() -> None:
    """No trial can synchronise in two steps; statistics must be NaN, not a crash."""
    (row,) = benchmark_sync(hidden_units=(3,), rules=("hebbian",), n=50, trials=2, max_steps=2)
    assert row.synchronized == 0
    assert row.success_rate == 0.0
    assert all(
        math.isnan(v)
        for v in (row.steps_mean, row.steps_median, row.steps_std, row.steps_min, row.steps_max)
    )


def test_sync_benchmark_reports_progress() -> None:
    calls: list[tuple[str, int, int]] = []
    benchmark_sync(
        hidden_units=(2, 3), rules=("hebbian",), n=8, trials=1, progress=lambda *a: calls.append(a)
    )
    assert calls == [("sync", 1, 2), ("sync", 2, 2)]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden_units": ()},
        {"hidden_units": (0,)},
        {"rules": ()},
        {"rules": ("oja",)},
        {"trials": 0},
        {"max_steps": 0},
    ],
)
def test_sync_benchmark_rejects_bad_arguments(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        benchmark_sync(**kwargs)


# --------------------------------------------------------------------- CSV
def test_write_csv_round_trips_channel_results(
    channel_results: list[ChannelBenchmarkResult], tmp_path: Path
) -> None:
    path = write_csv(channel_results, tmp_path / "nested" / "dir" / "channel.csv")
    rows = _read_csv(path)
    assert len(rows) == len(channel_results)
    assert list(rows[0]) == [
        "line_code", "snr_db", "trials", "bits_per_trial", "snr_in_db", "snr_out_db",
        "snr_gain_db", "ber_unfiltered", "ber_filtered", "bit_errors_unfiltered",
        "bit_errors_filtered",
    ]  # fmt: skip
    for parsed, original in zip(rows, channel_results):
        assert parsed["line_code"] == original.line_code
        assert float(parsed["snr_db"]) == original.snr_db
        assert int(parsed["bit_errors_filtered"]) == original.bit_errors_filtered
        assert float(parsed["snr_gain_db"]) == pytest.approx(original.snr_gain_db, rel=1e-5)
        assert float(parsed["ber_filtered"]) == pytest.approx(original.ber_filtered, rel=1e-5)


def test_write_csv_encodes_missing_statistics_as_nan(tmp_path: Path) -> None:
    rows = benchmark_sync(hidden_units=(3,), rules=("hebbian",), n=50, trials=1, max_steps=1)
    (parsed,) = _read_csv(write_csv(rows, tmp_path / "sync.csv"))
    assert math.isnan(float(parsed["steps_median"]))
    assert parsed["synchronized"] == "0"
    assert all(value != "" for value in parsed.values())


def test_write_csv_rejects_invalid_rows(
    channel_results: list[ChannelBenchmarkResult],
    sync_results: list[SyncBenchmarkResult],
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError):
        write_csv([], tmp_path / "empty.csv")
    with pytest.raises(ValueError):
        write_csv([channel_results[0], sync_results[0]], tmp_path / "mixed.csv")
    with pytest.raises(TypeError):
        write_csv([{"a": 1}], tmp_path / "dicts.csv")
    assert not list(tmp_path.iterdir())  # nothing half-written


# ------------------------------------------------------------------- plots
def test_plots_are_valid_png(
    channel_results: list[ChannelBenchmarkResult],
    sync_results: list[SyncBenchmarkResult],
    tmp_path: Path,
) -> None:
    for path in (
        plot_channel(channel_results, tmp_path / "channel.png"),
        plot_sync(sync_results, tmp_path / "sync.png"),
    ):
        data = path.read_bytes()
        assert data.startswith(PNG_MAGIC)
        assert len(data) > 5_000


def test_plots_handle_degenerate_results(tmp_path: Path) -> None:
    # A single point, with zero observed errors (log axis must not fail).
    single = benchmark_channel(snrs_db=(30.0,), line_codes=("manchester",), trials=1, bits_per_trial=64)
    assert single[0].ber_filtered == 0.0
    assert plot_channel(single, tmp_path / "single.png").stat().st_size > 0

    # No configuration synchronised: nothing to draw except the note.
    failed = benchmark_sync(hidden_units=(3,), rules=("hebbian",), n=50, trials=1, max_steps=1)
    assert plot_sync(failed, tmp_path / "failed.png").read_bytes().startswith(PNG_MAGIC)


def test_plots_reject_empty_results(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_channel([], tmp_path / "a.png")
    with pytest.raises(ValueError):
        plot_sync([], tmp_path / "b.png")


# ----------------------------------------------------------- orchestration
def test_run_benchmark_writes_all_artefacts(tmp_path: Path) -> None:
    out = tmp_path / "does" / "not" / "exist"
    report = run_benchmark(out, BenchmarkConfig.quick(seed=2))

    assert set(report.files) == {"channel_csv", "sync_csv", "channel_plot", "sync_plot"}
    assert sorted(p.name for p in out.iterdir()) == [
        "channel_benchmark.csv",
        "channel_benchmark.png",
        "tpm_sync_benchmark.csv",
        "tpm_sync_benchmark.png",
    ]
    channel_rows = _read_csv(report.files["channel_csv"])
    sync_rows = _read_csv(report.files["sync_csv"])
    assert len(channel_rows) == len(report.channel) == 2 * 5
    assert len(sync_rows) == len(report.sync) == 3 * 3
    assert {row["line_code"] for row in channel_rows} == set(LINE_CODES)
    assert {row["k"] for row in sync_rows} == {"3", "5", "7"}
    assert {row["rule"] for row in sync_rows} == set(LEARNING_RULES)
    for row in channel_rows:
        for column in ("snr_in_db", "snr_out_db", "snr_gain_db", "ber_filtered"):
            assert math.isfinite(float(row[column]))


def test_run_benchmark_without_plots_and_rerun(tmp_path: Path) -> None:
    config = BenchmarkConfig(
        snrs_db=(0.0,), channel_trials=1, bits_per_trial=100,
        hidden_units=(3,), rules=("hebbian",), tpm_n=8, sync_trials=1,
    )  # fmt: skip
    first = run_benchmark(tmp_path, config, make_plots=False)
    assert set(first.files) == {"channel_csv", "sync_csv"}
    assert not list(tmp_path.glob("*.png"))
    before = first.files["channel_csv"].read_text()
    run_benchmark(tmp_path, config, make_plots=False)  # overwrites cleanly
    assert first.files["channel_csv"].read_text() == before


def test_run_benchmark_rejects_file_as_output_dir(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x")
    with pytest.raises(NotADirectoryError):
        run_benchmark(target, BenchmarkConfig.quick())


def test_default_config_matches_specification() -> None:
    config = BenchmarkConfig()
    assert config.snrs_db == snr_grid(-5.0, 15.0, 2.5)
    assert config.line_codes == ("manchester", "nrz")
    assert config.hidden_units == (3, 5, 7)
    assert config.rules == LEARNING_RULES


# --------------------------------------------------------------------- CLI
def test_cli_benchmark_quick(tmp_path: Path) -> None:
    out = tmp_path / "bench"
    result = CliRunner().invoke(app, ["benchmark", "--quick", "--output-dir", str(out)])
    assert result.exit_code == 0, result.output
    assert "manchester" in result.output and "nrz" in result.output
    assert (out / "channel_benchmark.csv").is_file()
    assert (out / "tpm_sync_benchmark.csv").is_file()
    assert (out / "channel_benchmark.png").is_file()
    assert (out / "tpm_sync_benchmark.png").is_file()


def test_cli_benchmark_no_plots(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app, ["benchmark", "--quick", "--no-plots", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert not list(tmp_path.glob("*.png"))


def test_cli_benchmark_rejects_bad_options(tmp_path: Path) -> None:
    target = tmp_path / "file.txt"
    target.write_text("x")
    result = CliRunner().invoke(app, ["benchmark", "--quick", "--output-dir", str(target)])
    assert result.exit_code == 2


# --------------------------------------------------------- dashboard logic
def test_dashboard_channel_snapshot() -> None:
    from openbdn.app import simulate_channel

    snap = simulate_channel(snr_db=0.0, n_bits=800, seed=1)
    assert snap.clean.shape == snap.noisy.shape == snap.filtered.shape == snap.time_s.shape
    assert snap.snr_in_db == pytest.approx(0.0, abs=1e-6)
    assert snap.snr_gain_db > 5.0
    assert snap.window_snr_in_db.shape == snap.window_snr_out_db.shape == snap.window_time_s.shape
    assert snap.window_snr_in_db.size == 800 // 50
    assert np.all(np.diff(snap.window_time_s) > 0)


def test_dashboard_windowed_snr() -> None:
    from openbdn.app import windowed_snr_db

    ref = np.ones(10)
    out = windowed_snr_db(ref, ref + 0.1, window=4)
    assert out == pytest.approx([20.0, 20.0])  # partial window dropped
    assert np.isinf(windowed_snr_db(ref, ref, window=5)).all()
    assert windowed_snr_db(ref, ref, window=11).size == 0
    with pytest.raises(ValueError):
        windowed_snr_db(ref, ref, window=0)
    with pytest.raises(ValueError):
        windowed_snr_db(ref, ref[:5], window=2)


def test_dashboard_sync_trace() -> None:
    from openbdn.app import sync_trace

    seen: list[int] = []
    trace = sync_trace(k=3, n=16, seed=4, on_step=lambda step, _: seen.append(step))
    assert trace.result.synchronized
    assert trace.overlap.size == trace.result.steps + 1
    assert trace.overlap[-1] == 1.0 and trace.overlap[0] < 1.0
    assert seen == list(range(1, trace.result.steps + 1))

    short = sync_trace(k=3, n=100, seed=4, max_steps=3)
    assert not short.result.synchronized and short.overlap.size == 4


def test_dashboard_renders_without_exceptions() -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import openbdn.app as dashboard

    at = AppTest.from_file(dashboard.__file__, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "OpenBDN digital twin"
    labels = [m.label for m in at.metric]
    assert "SNR after NLMS" in labels and "GC-content" in labels

    # Move the SNR slider and switch line code: the app must re-render.
    at.slider(key="ch_snr").set_value(5.0)
    at.radio(key="ch_code").set_value("nrz")
    at.run()
    assert not at.exception

    # Run a small key exchange through the button.
    at.slider(key="tpm_n").set_value(16)
    at.button(key="tpm_run").click()
    at.run()
    assert not at.exception
    assert any("identical weights" in s.value for s in at.success)
