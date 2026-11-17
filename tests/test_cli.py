"""Tests for the end-to-end pipeline and the command line interface."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from openbdn import BDNPipeline, __version__
from openbdn.cli import app

runner = CliRunner()


# ------------------------------------------------------------------- pipeline
@pytest.mark.parametrize("message", ["A", "hello body", "BDN-04 ✓ 🧬", "x" * 200])
def test_pipeline_recovers_message_at_high_snr(message: str) -> None:
    report = BDNPipeline(seed=0).transmit(message, snr_db=20.0)
    assert report.success
    assert report.recovered_text == message
    assert report.received_dna == report.dna
    assert report.ber_filtered == 0.0
    assert report.gc_content == 0.5
    assert report.max_homopolymer <= 2
    assert report.channel_bits == 2 * len(report.dna)


def test_pipeline_equaliser_gains_snr() -> None:
    gains = [
        BDNPipeline(seed=seed).transmit("electro-quasistatic " * 20, snr_db=4.0).snr_gain_db
        for seed in range(5)
    ]
    assert min(gains) > 3.0
    assert sum(gains) / len(gains) > 8.0


def test_pipeline_reports_failure_without_raising() -> None:
    report = BDNPipeline(seed=1).transmit("this will not survive " * 10, snr_db=-45.0)
    assert not report.success
    assert report.ber_filtered > 0.0


def test_pipeline_works_with_nrz_line_code() -> None:
    from openbdn import EQChannelSimulator

    channel = EQChannelSimulator(line_code="nrz", seed=2)
    report = BDNPipeline(channel=channel, seed=2).transmit("nrz", snr_db=25.0)
    assert report.success


def test_pipeline_bit_text_helpers() -> None:
    bits = BDNPipeline.text_to_bits("hé")
    assert len(bits) == 24 and set(bits) <= {"0", "1"}
    assert BDNPipeline.bits_to_text(bits) == "hé"
    dna = "ACGTTGCA"
    assert BDNPipeline.channel_bits_to_dna(BDNPipeline.dna_to_channel_bits(dna)) == dna


def test_pipeline_rejects_empty_message() -> None:
    with pytest.raises(ValueError):
        BDNPipeline(seed=0).transmit("")


# ------------------------------------------------------------------------ CLI
def test_cli_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_simulate_sync() -> None:
    result = runner.invoke(app, ["simulate-sync", "--n", "32", "--seed", "1"])
    assert result.exit_code == 0, result.output
    assert "Synchronised" in result.output
    assert "Steps" in result.output
    assert "Shared 256-bit key" in result.output


def test_cli_simulate_sync_reports_timeout() -> None:
    result = runner.invoke(app, ["simulate-sync", "--max-steps", "3", "--seed", "1"])
    assert result.exit_code == 1
    assert "Not synchronised" in result.output


def test_cli_simulate_sync_rejects_bad_rule() -> None:
    result = runner.invoke(app, ["simulate-sync", "--rule", "nope"])
    assert result.exit_code == 2


def test_cli_transmit() -> None:
    result = runner.invoke(app, ["transmit", "hello BDN", "--snr-db", "15", "--seed", "3"])
    assert result.exit_code == 0, result.output
    assert "hello BDN" in result.output
    assert "GC-content" in result.output
    assert "50.0%" in result.output
    assert "SNR gain" in result.output
