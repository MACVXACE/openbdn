"""Constrained binary <-> DNA transcoding for synthetic DNA storage.

Synthesis and sequencing both degrade on unbalanced or repetitive strands, so
DNA storage codes bound the GC-content and the homopolymer run length. This
codec meets both limits *by construction* rather than by trial and
rejection, so they hold for every possible input.

Code construction
-----------------
Bits are taken three at a time and each triple is written as a dinucleotide
containing exactly one weak base (A/T) and one strong base (C/G):

====  ==========================  =====================
bit   meaning                     0 / 1
====  ==========================  =====================
b0    order within the pair       weak-strong / strong-weak
b1    which weak base             A / T
b2    which strong base           C / G
====  ==========================  =====================

Consequences:

* **GC-content is exactly 50 %** for every strand, and any window of ``w``
  bases contains between ``w/2 - 1`` and ``w/2 + 1`` G/C bases.
* **No homopolymer is longer than 2.** The two bases of a codeword differ,
  and any three consecutive positions contain a complete codeword.
* The information density is 1.5 bits per nucleotide.

A one-codeword header records how many zero bits (0-2) were appended to fill
the final triple, so inputs of any length round-trip exactly.
"""

from __future__ import annotations

__all__ = ["DNADecodeError", "DNAStorageCodec"]

_WEAK = "AT"
_STRONG = "CG"


class DNADecodeError(ValueError):
    """Raised when a DNA string is not a valid codeword sequence."""


def _build_tables() -> tuple[dict[str, str], dict[str, str]]:
    encode: dict[str, str] = {}
    for value in range(8):
        bits = format(value, "03b")
        weak = _WEAK[int(bits[1])]
        strong = _STRONG[int(bits[2])]
        encode[bits] = strong + weak if bits[0] == "1" else weak + strong
    decode = {pair: bits for bits, pair in encode.items()}
    return encode, decode


_ENCODE, _DECODE = _build_tables()


class DNAStorageCodec:
    """Lossless transcoder between binary strings and constrained DNA.

    Parameters
    ----------
    gc_min, gc_max:
        Admissible GC-content range used by :meth:`validate`.
    max_homopolymer:
        Longest admissible run of identical bases used by :meth:`validate`.

    Notes
    -----
    The code itself always produces 50 % GC and runs of at most 2, so any
    limits at least as loose as ``gc_min <= 0.5 <= gc_max`` and
    ``max_homopolymer >= 2`` are satisfied by every encoded strand. Tighter
    limits are rejected at construction time.
    """

    BITS_PER_CODEWORD = 3
    BASES_PER_CODEWORD = 2

    def __init__(
        self,
        gc_min: float = 0.40,
        gc_max: float = 0.60,
        max_homopolymer: int = 3,
    ) -> None:
        if not 0.0 <= gc_min <= 0.5 <= gc_max <= 1.0:
            raise ValueError("GC bounds must satisfy 0 <= gc_min <= 0.5 <= gc_max <= 1")
        if max_homopolymer < 2:
            raise ValueError("max_homopolymer must be >= 2 for this code")
        self.gc_min = float(gc_min)
        self.gc_max = float(gc_max)
        self.max_homopolymer = int(max_homopolymer)

    # ------------------------------------------------------------------
    def encode(self, binary_string: str) -> str:
        """Encode a string of ``'0'``/``'1'`` characters as DNA.

        Parameters
        ----------
        binary_string:
            Any-length (including empty) binary string.

        Returns
        -------
        str
            An ``ACGT`` strand of length ``2 * (1 + ceil(len / 3))`` with
            exactly 50 % GC-content and no homopolymer longer than 2.

        Raises
        ------
        TypeError
            If the input is not a string.
        ValueError
            If it contains characters other than ``'0'`` and ``'1'``.
        """
        if not isinstance(binary_string, str):
            raise TypeError("binary_string must be a str")
        if binary_string.strip("01"):
            raise ValueError("binary_string may contain only '0' and '1'")

        padding = -len(binary_string) % self.BITS_PER_CODEWORD
        payload = binary_string + "0" * padding
        codewords = [_ENCODE[format(padding, "03b")]]
        codewords.extend(
            _ENCODE[payload[i : i + 3]] for i in range(0, len(payload), 3)
        )
        return "".join(codewords)

    def decode(self, dna_string: str) -> str:
        """Decode a strand produced by :meth:`encode` back to binary.

        Surrounding whitespace and lower-case bases are accepted.

        Raises
        ------
        DNADecodeError
            If the strand has odd length, lacks a header, contains a
            dinucleotide that is not a codeword (which also catches every
            constraint-violating pair such as ``AA`` or ``GC``), or carries
            an inconsistent padding header.
        """
        if not isinstance(dna_string, str):
            raise TypeError("dna_string must be a str")
        strand = "".join(dna_string.split()).upper()
        if len(strand) < 2 or len(strand) % 2:
            raise DNADecodeError("strand length must be even and at least 2")

        triples: list[str] = []
        for i in range(0, len(strand), 2):
            pair = strand[i : i + 2]
            try:
                triples.append(_DECODE[pair])
            except KeyError:
                raise DNADecodeError(
                    f"invalid codeword {pair!r} at position {i}"
                ) from None

        padding = int(triples[0], 2)
        payload = "".join(triples[1:])
        if padding > 2:
            raise DNADecodeError("corrupt header: padding count out of range")
        if padding:
            if not payload or payload[-padding:] != "0" * padding:
                raise DNADecodeError("corrupt strand: padding bits are not zero")
            payload = payload[:-padding]
        return payload

    # ------------------------------------------------------------------
    @staticmethod
    def gc_content(dna_string: str) -> float:
        """Return the fraction of G and C bases (0.0 for an empty strand)."""
        if not dna_string:
            return 0.0
        strand = dna_string.upper()
        return (strand.count("G") + strand.count("C")) / len(strand)

    @staticmethod
    def max_homopolymer_run(dna_string: str) -> int:
        """Return the length of the longest run of identical bases."""
        longest = current = 0
        previous = ""
        for base in dna_string.upper():
            current = current + 1 if base == previous else 1
            previous = base
            longest = max(longest, current)
        return longest

    def validate(self, dna_string: str) -> bool:
        """Check a strand against the configured biological constraints."""
        strand = dna_string.upper()
        if not strand or strand.strip("ACGT"):
            return False
        return (
            self.gc_min <= self.gc_content(strand) <= self.gc_max
            and self.max_homopolymer_run(strand) <= self.max_homopolymer
        )
