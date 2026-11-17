"""Tests for Tree Parity Machine key agreement."""

from __future__ import annotations

import numpy as np
import pytest

from openbdn.crypto.vvtpm import VectorValuedTPM, synchronize


# ------------------------------------------------------------- architecture
def test_initial_weights_shape_and_bounds() -> None:
    tpm = VectorValuedTPM(k=4, n=25, l=5, seed=0)
    assert tpm.weights.shape == (4, 25)
    assert tpm.weights.min() >= -5 and tpm.weights.max() <= 5
    assert np.issubdtype(tpm.weights.dtype, np.integer)


def test_random_inputs_domain() -> None:
    tpm = VectorValuedTPM(k=3, n=200, l=3, m=4, seed=1)
    x = tpm.random_inputs()
    assert x.shape == (3, 200)
    assert set(np.unique(x)) == {-4, -3, -2, -1, 1, 2, 3, 4}


def test_binary_inputs_when_m_is_one() -> None:
    x = VectorValuedTPM(k=3, n=50, m=1, seed=2).random_inputs()
    assert set(np.unique(x)) == {-1, 1}


@pytest.mark.parametrize(
    "kwargs", [{"k": 0}, {"n": 0}, {"l": 0}, {"m": 0}, {"rule": "oja"}]
)
def test_constructor_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        VectorValuedTPM(**kwargs)


# ------------------------------------------------------------------ forward
def test_forward_matches_manual_computation() -> None:
    tpm = VectorValuedTPM(k=3, n=4, l=3, m=2, seed=0)
    tpm.weights = np.array([[1, -2, 3, 0], [0, 0, 0, 0], [-3, -3, 1, 2]])
    x = np.array([[1, 1, 1, 2], [2, -1, 1, -2], [1, 1, -1, -1]])
    # h = [2, 0, -9] -> sigma = [+1, -1 (zero field), -1] -> tau = +1
    assert tpm.forward(x) == 1
    np.testing.assert_array_equal(tpm.sigma, [1, -1, -1])


def test_forward_is_deterministic_and_binary() -> None:
    tpm = VectorValuedTPM(seed=3)
    rng = np.random.default_rng(3)
    for _ in range(50):
        x = tpm.random_inputs(rng)
        tau = tpm.forward(x)
        assert tau in (-1, 1)
        assert tpm.forward(x) == tau


@pytest.mark.parametrize(
    "bad",
    [
        np.ones((2, 10), dtype=int),  # wrong shape
        np.zeros((3, 10), dtype=int),  # zero components
        np.full((3, 10), 5, dtype=int),  # exceeds M
        np.ones((3, 10), dtype=float),  # non-integer dtype
    ],
)
def test_forward_rejects_invalid_inputs(bad: np.ndarray) -> None:
    with pytest.raises(ValueError):
        VectorValuedTPM(k=3, n=10, m=2, seed=0).forward(bad)


# ------------------------------------------------------------ learning rule
def test_no_update_when_outputs_disagree() -> None:
    tpm = VectorValuedTPM(k=3, n=20, seed=4)
    x = tpm.random_inputs()
    tau = tpm.forward(x)
    before = tpm.weights.copy()
    assert tpm.update_weights(x, tau, -tau) is False
    np.testing.assert_array_equal(tpm.weights, before)


def test_hebbian_update_only_touches_agreeing_units() -> None:
    tpm = VectorValuedTPM(k=3, n=30, l=4, m=2, seed=5)
    x = tpm.random_inputs()
    tau = tpm.forward(x)
    sigma = tpm.sigma.copy()
    before = tpm.weights.copy()

    assert tpm.update_weights(x, tau, tau) is True
    for unit in range(3):
        if sigma[unit] == tau:
            expected = np.clip(before[unit] + tau * x[unit], -4, 4)
        else:
            expected = before[unit]
        np.testing.assert_array_equal(tpm.weights[unit], expected)
    assert (sigma == tau).any()  # parity guarantees at least one active unit


@pytest.mark.parametrize(
    ("rule", "sign"), [("hebbian", 1), ("anti_hebbian", -1), ("random_walk", 0)]
)
def test_update_direction_per_rule(rule: str, sign: int) -> None:
    tpm = VectorValuedTPM(k=1, n=6, l=10, rule=rule, seed=6)  # type: ignore[arg-type]
    tpm.weights[:] = 0
    tpm.weights[0, 0] = 1  # ensures a non-zero field
    x = np.array([[1, -1, 1, -1, 1, -1]])
    tau = tpm.forward(x)
    before = tpm.weights.copy()
    tpm.update_weights(x, tau, tau)
    step = x if sign == 0 else sign * tau * x
    np.testing.assert_array_equal(tpm.weights, before + step)


def test_weights_stay_bounded() -> None:
    tpm = VectorValuedTPM(k=3, n=40, l=2, m=3, seed=7)
    rng = np.random.default_rng(7)
    for _ in range(500):
        x = tpm.random_inputs(rng)
        tau = tpm.forward(x)
        tpm.update_weights(x, tau, tau)
        assert np.abs(tpm.weights).max() <= 2


def test_update_rejects_bad_outputs() -> None:
    tpm = VectorValuedTPM(k=3, n=10, seed=8)
    x = tpm.random_inputs()
    tau = tpm.forward(x)
    with pytest.raises(ValueError):
        tpm.update_weights(x, 0, tau)
    with pytest.raises(ValueError):
        tpm.update_weights(x, -tau, -tau)  # inconsistent with own weights


# ---------------------------------------------------------- synchronisation
@pytest.mark.parametrize(
    ("k", "n", "l", "m"),
    [(3, 100, 3, 1), (3, 100, 3, 2), (3, 32, 4, 3), (4, 50, 3, 1), (2, 64, 5, 1)],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tpms_synchronise_within_epoch_limit(k: int, n: int, l: int, m: int, seed: int) -> None:
    alice = VectorValuedTPM(k, n, l, m, seed=seed)
    bob = VectorValuedTPM(k, n, l, m, seed=10_000 + seed)
    assert not alice.is_synchronized(bob)

    result = synchronize(alice, bob, max_steps=20_000, rng=seed)

    assert result.synchronized
    assert 0 < result.updates <= result.steps <= 20_000
    assert result.overlap == 1.0
    np.testing.assert_array_equal(alice.weights, bob.weights)
    assert result.key == alice.derive_key().hex() == bob.derive_key().hex()
    assert len(bytes.fromhex(result.key)) == 32


@pytest.mark.parametrize("rule", ["hebbian", "anti_hebbian", "random_walk"])
def test_all_learning_rules_synchronise(rule: str) -> None:
    alice = VectorValuedTPM(3, 64, 3, rule=rule, seed=11)  # type: ignore[arg-type]
    bob = VectorValuedTPM(3, 64, 3, rule=rule, seed=12)  # type: ignore[arg-type]
    assert synchronize(alice, bob, max_steps=20_000, rng=13).synchronized


def test_synchronised_machines_stay_synchronised() -> None:
    alice = VectorValuedTPM(3, 50, 3, seed=14)
    bob = VectorValuedTPM(3, 50, 3, seed=15)
    synchronize(alice, bob, rng=16)
    rng = np.random.default_rng(17)
    for _ in range(200):
        x = alice.random_inputs(rng)
        ta, tb = alice.forward(x), bob.forward(x)
        assert ta == tb
        alice.update_weights(x, ta, tb)
        bob.update_weights(x, tb, ta)
    assert alice.is_synchronized(bob)


def test_sync_reports_failure_when_budget_too_small() -> None:
    alice = VectorValuedTPM(3, 100, 3, seed=18)
    bob = VectorValuedTPM(3, 100, 3, seed=19)
    result = synchronize(alice, bob, max_steps=5, rng=20)
    assert not result.synchronized
    assert result.steps == 5
    assert result.key is None
    assert result.overlap < 1.0


def test_sync_callback_reports_progress() -> None:
    seen: list[tuple[int, float]] = []
    alice = VectorValuedTPM(3, 30, 3, seed=21)
    bob = VectorValuedTPM(3, 30, 3, seed=22)
    result = synchronize(alice, bob, rng=23, on_step=lambda s, o: seen.append((s, o)))
    assert [s for s, _ in seen] == list(range(1, result.steps + 1))
    assert seen[-1][1] == 1.0


def test_sync_rejects_mismatched_machines() -> None:
    with pytest.raises(ValueError):
        synchronize(VectorValuedTPM(3, 10), VectorValuedTPM(3, 11))
    with pytest.raises(ValueError):
        synchronize(VectorValuedTPM(3, 10), VectorValuedTPM(3, 10, rule="random_walk"))


def test_passive_observer_does_not_get_key_for_free() -> None:
    """An eavesdropper who only copies Alice's rule when *its* output agrees
    lags behind the two-way learners (sanity check, not a security proof)."""
    alice = VectorValuedTPM(3, 100, 3, seed=24)
    bob = VectorValuedTPM(3, 100, 3, seed=25)
    eve = VectorValuedTPM(3, 100, 3, seed=26)
    rng = np.random.default_rng(27)
    while not alice.is_synchronized(bob):
        x = alice.random_inputs(rng)
        ta, tb, te = alice.forward(x), bob.forward(x), eve.forward(x)
        if ta == tb == te:
            eve.update_weights(x, te, ta)
        alice.update_weights(x, ta, tb)
        bob.update_weights(x, tb, ta)
    assert not eve.is_synchronized(alice)


# --------------------------------------------------------------------- keys
def test_key_derivation() -> None:
    a = VectorValuedTPM(3, 20, 3, seed=28)
    b = VectorValuedTPM(3, 20, 3, seed=29)
    assert a.derive_key() == a.derive_key()
    assert a.derive_key() != b.derive_key()
    assert len(a.derive_key(16)) == 16
    assert len(a.derive_key(64)) == 64
    with pytest.raises(ValueError):
        a.derive_key(0)
