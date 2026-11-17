"""Tests for the constrained DNA storage codec."""

from __future__ import annotations

import itertools
import random

import pytest

from openbdn.storage.dna_codec import DNADecodeError, DNAStorageCodec


@pytest.fixture()
def codec() -> DNAStorageCodec:
    return DNAStorageCodec()


def _assert_constraints(codec: DNAStorageCodec, dna: str) -> None:
    assert set(dna) <= set("ACGT")
    assert 0.40 <= codec.gc_content(dna) <= 0.60
    assert codec.max_homopolymer_run(dna) <= 3
    assert codec.validate(dna)


# ------------------------------------------------------------------ roundtrip
@pytest.mark.parametrize(
    "bits",
    [
        "",
        "0",
        "1",
        "01",
        "000",
        "111",
        "0000000000000000",
        "1111111111111111",
        "0101010101010101",
        "0011001100110011",
        "000111000111000111",
        "0" * 1000,
        "1" * 1001,
    ],
)
def test_roundtrip_edge_cases(codec: DNAStorageCodec, bits: str) -> None:
    dna = codec.encode(bits)
    assert codec.decode(dna) == bits
    _assert_constraints(codec, dna)


def test_roundtrip_exhaustive_short_strings(codec: DNAStorageCodec) -> None:
    """Every binary string of length 0..12 (8191 strings) must round-trip."""
    seen: set[str] = set()
    for length in range(13):
        for combo in itertools.product("01", repeat=length):
            bits = "".join(combo)
            dna = codec.encode(bits)
            assert codec.decode(dna) == bits
            _assert_constraints(codec, dna)
            seen.add(dna)
    assert len(seen) == 2**13 - 1  # encoding is injective


def test_roundtrip_random_fuzz(codec: DNAStorageCodec) -> None:
    rng = random.Random(1234)
    for _ in range(2000):
        bits = "".join(rng.choice("01") for _ in range(rng.randrange(0, 400)))
        dna = codec.encode(bits)
        assert codec.decode(dna) == bits
        _assert_constraints(codec, dna)


def test_roundtrip_utf8_payload(codec: DNAStorageCodec) -> None:
    text = "BDN-04: données biologiques 🧬"
    bits = "".join(format(b, "08b") for b in text.encode())
    recovered = codec.decode(codec.encode(bits))
    data = bytes(int(recovered[i : i + 8], 2) for i in range(0, len(recovered), 8))
    assert data.decode() == text


# ---------------------------------------------------------------- constraints
def test_gc_content_is_exactly_half(codec: DNAStorageCodec) -> None:
    rng = random.Random(5)
    for _ in range(200):
        bits = "".join(rng.choice("01") for _ in range(rng.randrange(1, 300)))
        assert codec.gc_content(codec.encode(bits)) == 0.5


def test_homopolymers_never_exceed_two(codec: DNAStorageCodec) -> None:
    rng = random.Random(6)
    worst = 0
    for _ in range(500):
        bits = "".join(rng.choice("01") for _ in range(300))
        worst = max(worst, codec.max_homopolymer_run(codec.encode(bits)))
    assert worst == 2


def test_local_gc_window_is_balanced(codec: DNAStorageCodec) -> None:
    """GC balance holds locally, not just globally (any 20-nt window)."""
    rng = random.Random(7)
    dna = codec.encode("".join(rng.choice("01") for _ in range(3000)))
    for start in range(len(dna) - 20):
        window = dna[start : start + 20]
        assert 0.40 <= codec.gc_content(window) <= 0.60


def test_strand_length_formula(codec: DNAStorageCodec) -> None:
    for length in range(0, 50):
        expected = 2 * (1 + -(-length // 3))
        assert len(codec.encode("0" * length)) == expected


def test_encoding_is_deterministic(codec: DNAStorageCodec) -> None:
    assert codec.encode("010011") == codec.encode("010011")
    # header (pad 0 -> 000 -> AC), 010 -> TC, 011 -> TG
    assert codec.encode("010011") == "ACTCTG"


# ------------------------------------------------------------------- failures
@pytest.mark.parametrize("bad", ["012", "abc", "0 1", "１０"])
def test_encode_rejects_non_binary(codec: DNAStorageCodec, bad: str) -> None:
    with pytest.raises(ValueError):
        codec.encode(bad)


def test_encode_rejects_non_string(codec: DNAStorageCodec) -> None:
    with pytest.raises(TypeError):
        codec.encode(1010)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad",
    [
        "",  # no header
        "A",  # odd length
        "ACT",  # odd length
        "AA",  # homopolymer pair is not a codeword
        "ACGC",  # strong-strong pair
        "ACAT",  # weak-weak pair
        "ACNX",  # non-nucleotide characters
        "GTAC",  # header claims padding 7
        "AGTG",  # header claims 1 pad bit but last payload bit is 1
        "AG",  # header claims padding with empty payload
    ],
)
def test_decode_rejects_corrupt_strands(codec: DNAStorageCodec, bad: str) -> None:
    with pytest.raises(DNADecodeError):
        codec.decode(bad)


def test_decode_tolerates_case_and_whitespace(codec: DNAStorageCodec) -> None:
    dna = codec.encode("1011001")
    assert codec.decode(f"  {dna.lower()[:4]}\n{dna[4:]} ") == "1011001"


def test_single_base_substitutions_never_decode_silently_to_same_data(
    codec: DNAStorageCodec,
) -> None:
    bits = "110100101110001011"
    dna = codec.encode(bits)
    for i, original in enumerate(dna):
        for base in "ACGT":
            if base == original:
                continue
            mutated = dna[:i] + base + dna[i + 1 :]
            try:
                assert codec.decode(mutated) != bits
            except DNADecodeError:
                pass


# -------------------------------------------------------------------- helpers
def test_metric_helpers() -> None:
    assert DNAStorageCodec.gc_content("") == 0.0
    assert DNAStorageCodec.gc_content("GGCC") == 1.0
    assert DNAStorageCodec.gc_content("acgt") == 0.5
    assert DNAStorageCodec.max_homopolymer_run("") == 0
    assert DNAStorageCodec.max_homopolymer_run("ACGT") == 1
    assert DNAStorageCodec.max_homopolymer_run("AACCCCGT") == 4


def test_validate_flags_violations(codec: DNAStorageCodec) -> None:
    assert not codec.validate("")
    assert not codec.validate("AAAATTTT")  # 0 % GC
    assert not codec.validate("ACGGGGTA")  # homopolymer of 4
    assert not codec.validate("ACGU")
    assert codec.validate("ACGTACGT")


def test_constructor_rejects_unsatisfiable_limits() -> None:
    with pytest.raises(ValueError):
        DNAStorageCodec(gc_min=0.55, gc_max=0.60)
    with pytest.raises(ValueError):
        DNAStorageCodec(max_homopolymer=1)
