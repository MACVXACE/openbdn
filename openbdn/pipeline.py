"""End-to-end BDN-04 transmission pipeline.

``text -> bits -> constrained DNA -> line-coded waveform -> EMG-corrupted EQ-HBC
channel -> NLMS equaliser -> detector -> DNA -> bits -> text``
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from openbdn.channel.eq_hbc import EQChannelSimulator
from openbdn.storage.dna_codec import DNADecodeError, DNAStorageCodec

__all__ = ["BDNPipeline", "TransmissionReport"]

_BASE_TO_BITS = {"A": (0, 0), "C": (0, 1), "G": (1, 0), "T": (1, 1)}
_BITS_TO_BASE = {bits: base for base, bits in _BASE_TO_BITS.items()}


@dataclass(frozen=True)
class TransmissionReport:
    """Result of one :meth:`BDNPipeline.transmit` call.

    Attributes
    ----------
    message:
        The text that was sent.
    dna:
        The encoded strand that was put on the channel.
    gc_content:
        GC fraction of ``dna``.
    max_homopolymer:
        Longest homopolymer run in ``dna``.
    channel_bits:
        Number of payload bits sent over the channel (2 per nucleotide).
    snr_in_db:
        Payload SNR at the receiver input.
    snr_out_db:
        Payload SNR after NLMS equalisation.
    ber_unfiltered:
        Payload bit error rate when detecting the raw received waveform.
    ber_filtered:
        Payload bit error rate after equalisation.
    received_dna:
        The strand reconstructed by the receiver.
    recovered_text:
        Decoded text, or ``None`` if the received strand was not decodable.
    error:
        Decoder error message when ``recovered_text`` is ``None``.
    """

    message: str
    dna: str
    gc_content: float
    max_homopolymer: int
    channel_bits: int
    snr_in_db: float
    snr_out_db: float
    ber_unfiltered: float
    ber_filtered: float
    received_dna: str
    recovered_text: str | None
    error: str | None = None

    @property
    def success(self) -> bool:
        """``True`` when the recovered text equals the original message."""
        return self.recovered_text == self.message

    @property
    def snr_gain_db(self) -> float:
        """SNR improvement delivered by the equaliser."""
        return self.snr_out_db - self.snr_in_db


class BDNPipeline:
    """Encode text as DNA, send it through the EQ-HBC channel and decode it.

    Parameters
    ----------
    channel:
        Channel simulator; a default one is created if omitted.
    codec:
        DNA codec; a default one is created if omitted.
    preamble_bits:
        Length of the pseudo-random training preamble sent ahead of the
        payload. The receiver knows the preamble and uses it to train the
        equaliser before switching to decision-directed adaptation.
    seed:
        Seed for the default channel and the preamble.
    """

    def __init__(
        self,
        channel: EQChannelSimulator | None = None,
        codec: DNAStorageCodec | None = None,
        preamble_bits: int = 256,
        seed: int | None = None,
    ) -> None:
        if preamble_bits < 0:
            raise ValueError("preamble_bits must be non-negative")
        self.channel = channel if channel is not None else EQChannelSimulator(seed=seed)
        self.codec = codec if codec is not None else DNAStorageCodec()
        self.preamble_bits = int(preamble_bits)
        self._preamble = np.random.default_rng(seed).integers(
            0, 2, size=self.preamble_bits, dtype=np.uint8
        )

    # -- conversions ----------------------------------------------------
    @staticmethod
    def text_to_bits(text: str) -> str:
        """UTF-8 encode ``text`` and return it as a binary string."""
        return "".join(format(byte, "08b") for byte in text.encode("utf-8"))

    @staticmethod
    def bits_to_text(bits: str) -> str:
        """Inverse of :meth:`text_to_bits` (invalid UTF-8 is replaced)."""
        usable = len(bits) - len(bits) % 8
        data = bytes(int(bits[i : i + 8], 2) for i in range(0, usable, 8))
        return data.decode("utf-8", errors="replace")

    @staticmethod
    def dna_to_channel_bits(dna: str) -> np.ndarray:
        """Map each nucleotide to two channel bits (A=00, C=01, G=10, T=11)."""
        flat = [bit for base in dna for bit in _BASE_TO_BITS[base]]
        return np.array(flat, dtype=np.uint8)

    @staticmethod
    def channel_bits_to_dna(bits: np.ndarray) -> str:
        """Inverse of :meth:`dna_to_channel_bits`."""
        pairs = np.asarray(bits, dtype=np.uint8).reshape(-1, 2)
        return "".join(_BITS_TO_BASE[(int(a), int(b))] for a, b in pairs)

    # -- transmission ---------------------------------------------------
    def transmit(self, message: str, snr_db: float = 5.0) -> TransmissionReport:
        """Send ``message`` across the simulated link.

        Parameters
        ----------
        message:
            Non-empty text to transmit.
        snr_db:
            Channel SNR (signal power over EMG power) in decibels.

        Returns
        -------
        TransmissionReport
        """
        if not message:
            raise ValueError("message must not be empty")

        dna = self.codec.encode(self.text_to_bits(message))
        payload = self.dna_to_channel_bits(dna)
        frame = np.concatenate((self._preamble, payload))

        ch = self.channel
        clean = ch.modulate(frame)
        received = ch.add_muscle_noise(clean, snr_db)
        n_train = self.preamble_bits * ch.samples_per_symbol
        equalised = ch.apply_adaptive_filter(received, training=clean[:n_train])

        body = slice(n_train, None)
        rx_raw = ch.demodulate(received[body])
        rx_eq = ch.demodulate(equalised[body])
        received_dna = self.channel_bits_to_dna(rx_eq)

        try:
            recovered: str | None = self.bits_to_text(self.codec.decode(received_dna))
            error = None
        except DNADecodeError as exc:
            recovered, error = None, str(exc)

        return TransmissionReport(
            message=message,
            dna=dna,
            gc_content=self.codec.gc_content(dna),
            max_homopolymer=self.codec.max_homopolymer_run(dna),
            channel_bits=int(payload.size),
            snr_in_db=ch.snr_db(clean[body], received[body]),
            snr_out_db=ch.snr_db(clean[body], equalised[body]),
            ber_unfiltered=float(np.mean(rx_raw != payload)),
            ber_filtered=float(np.mean(rx_eq != payload)),
            received_dna=received_dna,
            recovered_text=recovered,
            error=error,
        )
