"""Small, dependency-free early-stopping controller for round metrics.

The controller counts *evaluation events* rather than optimizer steps.  A
patience of zero disables stopping, which preserves the historical fixed-round
protocol unless a run opts in explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EarlyStopDecision:
    """Decision returned after one evaluated communication round."""

    stop: bool
    improved: bool
    best_value: Optional[float]
    best_round: Optional[int]
    bad_evaluations: int
    reason: str


class RoundEarlyStopping:
    """Patience-based stopping for a scalar metric observed per round.

    ``mode='max'`` is appropriate for accuracy; ``mode='min'`` is appropriate
    for loss.  ``min_rounds`` is a warm-up in communication rounds.  Lack of
    improvement during warm-up never contributes to the patience counter.
    ``min_delta`` is expressed in the same units as the metric (percentage
    points for the logged accuracy values).
    """

    def __init__(
        self,
        *,
        patience: int = 0,
        min_delta: float = 0.0,
        min_rounds: int = 0,
        mode: str = "max",
    ) -> None:
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.min_rounds = int(min_rounds)
        self.mode = str(mode).lower().strip()
        if self.patience < 0:
            raise ValueError("patience must be non-negative")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative")
        if self.min_rounds < 0:
            raise ValueError("min_rounds must be non-negative")
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.best_value: Optional[float] = None
        self.best_round: Optional[int] = None
        self.bad_evaluations = 0

    @property
    def enabled(self) -> bool:
        return self.patience > 0

    def _is_improvement(self, value: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "max":
            return float(value) > float(self.best_value) + self.min_delta
        return float(value) < float(self.best_value) - self.min_delta

    def update(self, value: float, round_num: int) -> EarlyStopDecision:
        """Consume one evaluated round and return the stopping decision."""

        value = float(value)
        round_num = int(round_num)
        improved = self._is_improvement(value)

        if improved:
            self.best_value = value
            self.best_round = round_num
            self.bad_evaluations = 0
            return EarlyStopDecision(
                stop=False,
                improved=True,
                best_value=self.best_value,
                best_round=self.best_round,
                bad_evaluations=self.bad_evaluations,
                reason="improved",
            )

        if not self.enabled or round_num < self.min_rounds:
            # Do not consume patience before the warm-up or when disabled.
            self.bad_evaluations = 0 if round_num < self.min_rounds else self.bad_evaluations
            return EarlyStopDecision(
                stop=False,
                improved=False,
                best_value=self.best_value,
                best_round=self.best_round,
                bad_evaluations=self.bad_evaluations,
                reason="disabled" if not self.enabled else "warmup",
            )

        self.bad_evaluations += 1
        stop = self.bad_evaluations >= self.patience
        return EarlyStopDecision(
            stop=stop,
            improved=False,
            best_value=self.best_value,
            best_round=self.best_round,
            bad_evaluations=self.bad_evaluations,
            reason="patience_exhausted" if stop else "no_improvement",
        )
