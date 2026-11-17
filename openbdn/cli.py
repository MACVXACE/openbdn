"""Command line interface for OpenBDN.

Run ``openbdn --help`` for the list of commands.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from openbdn import __version__
from openbdn.channel.eq_hbc import EQChannelSimulator
from openbdn.crypto.vvtpm import VectorValuedTPM, synchronize
from openbdn.pipeline import BDNPipeline

app = typer.Typer(
    name="openbdn",
    help="OpenBDN: simulate a BDN-04 Biological Data Network link.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"openbdn {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """OpenBDN: simulate a BDN-04 Biological Data Network link."""


@app.command("simulate-sync")
def simulate_sync(
    k: int = typer.Option(3, "--k", "-k", min=1, help="Hidden units K."),
    n: int = typer.Option(100, "--n", "-n", min=1, help="Inputs per hidden unit N."),
    l: int = typer.Option(3, "--l", "-l", min=1, help="Synaptic depth L."),  # noqa: E741
    m: int = typer.Option(1, "--m", "-m", min=1, help="Input magnitude bound M."),
    rule: str = typer.Option(
        "hebbian", "--rule", help="hebbian, anti_hebbian or random_walk."
    ),
    max_steps: int = typer.Option(
        100_000, "--max-steps", min=1, help="Give up after this many steps."
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed for reproducibility."),
) -> None:
    """Synchronise two Tree Parity Machines (Alice and Bob) over a public channel."""
    seeds = np.random.SeedSequence(seed).spawn(3)
    try:
        alice = VectorValuedTPM(k, n, l, m, rule=rule, seed=np.random.default_rng(seeds[0]))  # type: ignore[arg-type]
        bob = VectorValuedTPM(k, n, l, m, rule=rule, seed=np.random.default_rng(seeds[1]))  # type: ignore[arg-type]
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=2) from None

    console.print(
        Panel.fit(
            f"K={k}  N={n}  L={l}  M={m}  rule={rule}\n"
            f"initial weight overlap: {alice.overlap(bob):.1%}",
            title="VV-TPM key agreement",
            border_style="cyan",
        )
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]weight overlap"),
        BarColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        TextColumn("step {task.fields[step]}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("sync", total=1.0, step=0)

        def on_step(step: int, overlap: float) -> None:
            if step % 16 == 0:
                progress.update(task, completed=overlap, step=step)

        result = synchronize(
            alice, bob, max_steps=max_steps, rng=np.random.default_rng(seeds[2]), on_step=on_step
        )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Synchronised", "[green]yes[/]" if result.synchronized else "[red]no[/]")
    table.add_row("Steps (inputs exchanged)", f"{result.steps:,}")
    table.add_row("Weight updates", f"{result.updates:,}")
    table.add_row("Final weight overlap", f"{result.overlap:.1%}")
    console.print(table)
    if result.key is not None:
        console.print("  Shared 256-bit key:")
        console.print(f"  [bold green]{result.key}[/]", soft_wrap=True)

    if not result.synchronized:
        console.print(f"[yellow]Not synchronised within {max_steps:,} steps.[/]")
        raise typer.Exit(code=1)


@app.command("transmit")
def transmit(
    message: str = typer.Argument(..., help="Text to transmit."),
    snr_db: float = typer.Option(5.0, "--snr-db", help="Channel SNR in dB."),
    line_code: str = typer.Option("manchester", "--line-code", help="manchester or nrz."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Seed for reproducibility."),
    taps: int = typer.Option(33, "--taps", min=1, help="NLMS filter length."),
    step_size: float = typer.Option(0.05, "--step-size", help="NLMS step size mu."),
) -> None:
    """Encode text as DNA, send it over the EQ-HBC channel and decode it."""
    try:
        channel = EQChannelSimulator(
            line_code=line_code,  # type: ignore[arg-type]
            num_taps=taps,
            step_size=step_size,
            seed=seed,
        )
        report = BDNPipeline(channel=channel, seed=seed).transmit(message, snr_db=snr_db)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise typer.Exit(code=2) from None

    storage = Table(title="DNA storage layer", show_header=False, title_justify="left")
    storage.add_column(no_wrap=True)
    storage.add_column(overflow="fold")
    storage.add_row("Strand", report.dna)
    storage.add_row("Length", f"{len(report.dna)} nt")
    storage.add_row("GC-content", f"{report.gc_content:.1%}")
    storage.add_row("Longest homopolymer", str(report.max_homopolymer))
    console.print(storage)

    link = Table(title="EQ-HBC channel", title_justify="left")
    link.add_column("Metric")
    link.add_column("Before NLMS", justify="right")
    link.add_column("After NLMS", justify="right")
    link.add_row("SNR", f"{report.snr_in_db:.2f} dB", f"{report.snr_out_db:.2f} dB")
    link.add_row(
        "Bit error rate", f"{report.ber_unfiltered:.4f}", f"{report.ber_filtered:.4f}"
    )
    console.print(link)
    console.print(
        f"SNR gain: [bold]{report.snr_gain_db:+.2f} dB[/] over {report.channel_bits} payload bits"
    )

    if report.success:
        console.print(
            Panel(report.recovered_text or "", title="Recovered message", border_style="green")
        )
        return

    detail = report.error or "decoded text differs from the original"
    shown = report.recovered_text if report.recovered_text is not None else "(undecodable strand)"
    console.print(
        Panel(f"{shown}\n\n[dim]{detail}[/]", title="Transmission failed", border_style="red")
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":  # pragma: no cover
    app()
