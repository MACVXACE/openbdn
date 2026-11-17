"""Neural key agreement with vector-valued Tree Parity Machines.

A Tree Parity Machine (TPM) is a two-layer feed-forward network with ``K``
hidden perceptrons, each connected to its own ``N`` inputs through bounded
integer weights. Two parties (Alice and Bob) holding independently
initialised TPMs can drive their weights to the same values by exchanging
only their *output bits* for a stream of public random inputs; the common
weights are then hashed into a symmetric key.

This implementation generalises the binary TPM to integer-valued input
vectors with components in ``{-M, ..., -1, 1, ..., M}``. ``M = 1`` recovers
the classical machine of Kanter, Kinzel and Kanter (2002).

.. warning::
   Neural key exchange is a research construction. Geometric, genetic and
   majority attacks are known against TPM synchronisation, and it carries no
   reduction to a standard hardness assumption. Do not use it to protect
   real data.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

IntArray = NDArray[np.int64]
LearningRule = Literal["hebbian", "anti_hebbian", "random_walk"]

__all__ = ["LearningRule", "SyncResult", "VectorValuedTPM", "synchronize"]

_RULES: tuple[str, ...] = ("hebbian", "anti_hebbian", "random_walk")


class VectorValuedTPM:
    r"""A Tree Parity Machine with bounded integer weights and inputs.

    Architecture
    ------------
    * ``K`` hidden units, each with ``N`` inputs (``K * N`` input neurons).
    * Weights :math:`w_{kj} \in \{-L, \dots, L\}`.
    * Inputs :math:`x_{kj} \in \{-M, \dots, -1, 1, \dots, M\}`.

    The hidden-unit and network outputs are

    .. math::

        h_k = \sum_{j=1}^{N} w_{kj} x_{kj}, \qquad
        \sigma_k = \begin{cases} +1 & h_k > 0 \\ -1 & h_k \le 0 \end{cases},
        \qquad
        \tau = \prod_{k=1}^{K} \sigma_k .

    Parameters
    ----------
    k:
        Number of hidden units ``K``.
    n:
        Number of inputs per hidden unit ``N``.
    l:
        Synaptic depth ``L`` (weights are clipped to ``[-L, L]``).
    m:
        Input magnitude bound ``M``.
    rule:
        Learning rule: ``"hebbian"``, ``"anti_hebbian"`` or ``"random_walk"``.
    seed:
        Seed or generator used for the secret initial weights.
    """

    def __init__(
        self,
        k: int = 3,
        n: int = 100,
        l: int = 3,  # noqa: E741 - conventional symbol for synaptic depth
        m: int = 1,
        rule: LearningRule = "hebbian",
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if k < 1 or n < 1:
            raise ValueError("k and n must be positive integers")
        if l < 1:
            raise ValueError("synaptic depth l must be >= 1")
        if m < 1:
            raise ValueError("input bound m must be >= 1")
        if rule not in _RULES:
            raise ValueError(f"rule must be one of {_RULES}, got {rule!r}")

        self.k = int(k)
        self.n = int(n)
        self.l = int(l)
        self.m = int(m)
        self.rule: LearningRule = rule
        self._rng = np.random.default_rng(seed)
        #: Secret weight matrix of shape ``(K, N)``.
        self.weights: IntArray = self._rng.integers(
            -self.l, self.l + 1, size=(self.k, self.n), dtype=np.int64
        )
        #: Hidden-unit outputs from the most recent :meth:`forward` call.
        self.sigma: IntArray | None = None

    # ------------------------------------------------------------------
    def random_inputs(self, rng: np.random.Generator | None = None) -> IntArray:
        """Draw a public input matrix of shape ``(K, N)``.

        Components are uniform on ``{-M, ..., -1, 1, ..., M}``.
        """
        gen = self._rng if rng is None else rng
        magnitude = gen.integers(1, self.m + 1, size=(self.k, self.n), dtype=np.int64)
        sign = gen.integers(0, 2, size=(self.k, self.n), dtype=np.int64) * 2 - 1
        return magnitude * sign

    def forward(self, inputs: ArrayLike) -> int:
        """Compute the output bit ``tau`` for a public input.

        Parameters
        ----------
        inputs:
            Integer array of shape ``(K, N)`` with non-zero components of
            magnitude at most ``M``.

        Returns
        -------
        int
            The parity output, ``+1`` or ``-1``. The hidden-unit outputs are
            stored in :attr:`sigma`.
        """
        x = self._validate_inputs(inputs)
        self.sigma = self._hidden_outputs(x)
        return int(np.prod(self.sigma))

    def update_weights(self, inputs: ArrayLike, output: int, sync_output: int) -> bool:
        r"""Apply one step of the mutual learning rule.

        Weights change only when both parties produced the same output
        (:math:`\tau^A = \tau^B`), and then only in hidden units that agree
        with that output (:math:`\sigma_k = \tau`):

        .. math::

            w_{kj} \leftarrow g\!\left(w_{kj} + \Delta_{kj}\,
                \Theta(\sigma_k \tau)\,\Theta(\tau^A \tau^B)\right),

        with :math:`g` clipping to ``[-L, L]`` and

        * Hebbian: :math:`\Delta_{kj} = \tau\, x_{kj}`
        * anti-Hebbian: :math:`\Delta_{kj} = -\tau\, x_{kj}`
        * random walk: :math:`\Delta_{kj} = x_{kj}`

        Parameters
        ----------
        inputs:
            The public input used for this round.
        output:
            This machine's output ``tau`` for ``inputs``.
        sync_output:
            The partner's output for the same ``inputs``.

        Returns
        -------
        bool
            ``True`` if the weights were updated, ``False`` if the outputs
            disagreed and the step was skipped.

        Raises
        ------
        ValueError
            If an output is not ``+1``/``-1`` or ``output`` is inconsistent
            with this machine's weights and ``inputs``.
        """
        if output not in (-1, 1) or sync_output not in (-1, 1):
            raise ValueError("outputs must be +1 or -1")
        x = self._validate_inputs(inputs)
        sigma = self._hidden_outputs(x)
        if int(np.prod(sigma)) != output:
            raise ValueError("output does not match this machine's response to inputs")
        if output != sync_output:
            return False

        if self.rule == "hebbian":
            delta = output * x
        elif self.rule == "anti_hebbian":
            delta = -output * x
        else:  # random_walk
            delta = x

        active = sigma == output
        self.weights[active] = np.clip(
            self.weights[active] + delta[active], -self.l, self.l
        )
        return True

    # ------------------------------------------------------------------
    def overlap(self, other: "VectorValuedTPM") -> float:
        """Return the fraction of weights identical to those of ``other``."""
        if self.weights.shape != other.weights.shape:
            raise ValueError("machines have different architectures")
        return float(np.mean(self.weights == other.weights))

    def is_synchronized(self, other: "VectorValuedTPM") -> bool:
        """Return ``True`` when every weight equals the partner's."""
        return self.weights.shape == other.weights.shape and bool(
            np.array_equal(self.weights, other.weights)
        )

    def derive_key(self, length: int = 32) -> bytes:
        """Derive ``length`` key bytes from the weights with SHAKE-256.

        The architecture parameters are bound into the hash so that machines
        with different shapes can never produce the same key.
        """
        if length < 1:
            raise ValueError("length must be >= 1")
        header = f"openbdn-tpm|{self.k}|{self.n}|{self.l}|{self.m}|".encode()
        body = (self.weights + self.l).astype("<u4").tobytes()
        return hashlib.shake_256(header + body).digest(length)

    # ------------------------------------------------------------------
    def _hidden_outputs(self, x: IntArray) -> IntArray:
        fields = np.einsum("kn,kn->k", self.weights, x)
        return np.where(fields > 0, 1, -1).astype(np.int64)

    def _validate_inputs(self, inputs: ArrayLike) -> IntArray:
        raw = np.asarray(inputs)
        if raw.shape != (self.k, self.n):
            raise ValueError(
                f"inputs must have shape ({self.k}, {self.n}), got {raw.shape}"
            )
        if not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("inputs must be integers")
        x = raw.astype(np.int64, copy=False)
        magnitude = np.abs(x)
        if magnitude.min() < 1 or magnitude.max() > self.m:
            raise ValueError(f"input components must satisfy 1 <= |x| <= {self.m}")
        return x

    def __repr__(self) -> str:
        return (
            f"VectorValuedTPM(k={self.k}, n={self.n}, l={self.l}, "
            f"m={self.m}, rule={self.rule!r})"
        )


@dataclass(frozen=True)
class SyncResult:
    """Outcome of a :func:`synchronize` run.

    Attributes
    ----------
    synchronized:
        Whether full weight equality was reached.
    steps:
        Number of public inputs exchanged.
    updates:
        Number of those steps in which the outputs agreed and weights moved.
    overlap:
        Final fraction of identical weights.
    key:
        Shared key (hex) if synchronised, otherwise ``None``.
    """

    synchronized: bool
    steps: int
    updates: int
    overlap: float
    key: str | None


def synchronize(
    alice: VectorValuedTPM,
    bob: VectorValuedTPM,
    max_steps: int = 100_000,
    rng: np.random.Generator | int | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> SyncResult:
    """Run the mutual-learning protocol until Alice and Bob share weights.

    Each step draws one public input, both parties publish their output bit,
    and both apply :meth:`VectorValuedTPM.update_weights`.

    Parameters
    ----------
    alice, bob:
        Machines with identical architecture and learning rule. They are
        modified in place.
    max_steps:
        Upper bound on the number of exchanged inputs.
    rng:
        Source of the *public* inputs.
    on_step:
        Optional callback ``on_step(step, overlap)`` invoked after each step,
        e.g. to drive a progress display.

    Returns
    -------
    SyncResult
    """
    signature = lambda t: (t.k, t.n, t.l, t.m, t.rule)  # noqa: E731
    if signature(alice) != signature(bob):
        raise ValueError("alice and bob must share architecture and learning rule")
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    public = np.random.default_rng(rng)
    steps = 0
    updates = 0
    while steps < max_steps and not alice.is_synchronized(bob):
        x = alice.random_inputs(public)
        tau_a = alice.forward(x)
        tau_b = bob.forward(x)
        moved = alice.update_weights(x, tau_a, tau_b)
        bob.update_weights(x, tau_b, tau_a)
        updates += int(moved)
        steps += 1
        if on_step is not None:
            on_step(steps, alice.overlap(bob))

    done = alice.is_synchronized(bob)
    return SyncResult(
        synchronized=done,
        steps=steps,
        updates=updates,
        overlap=alice.overlap(bob),
        key=alice.derive_key().hex() if done else None,
    )
