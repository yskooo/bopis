"""The BOPIS Bayesian Optimization engine and the random-search baseline.

Implements Chapter 3's six-step procedure verbatim:

1. **Initialize** -- draw ``n_seeds`` configurations from ``X_feasible`` using the
   task-informed variant prior ``P(p | task)``; evaluate each on the proxy
   prompt subset.
2. **Fit GP surrogate** -- fit a Gaussian Process to all observed
   ``{x_i, E(x_i)}`` pairs, yielding ``mu(x)`` and ``sigma(x)``.
3. **Acquire next config** -- select ``x_{n+1} = argmax EI(x)`` over
   ``X_feasible``.
4. **Evaluate** -- run inference under ``x_{n+1}``; record ``E``, ``S``, ``Q``.
5. **Update** -- append the observation, refit the GP posterior, return to 3.
6. **Terminate** -- after ``N`` total iterations
   (``n_seeds`` random + ``N - n_seeds`` BO-guided); hand all evaluations to
   Pareto analysis.

The ``mu`` and ``sigma`` recorded at each BO-guided iteration are genuine
*one-step-ahead* predictions: they come from the GP fitted on the previous
iterations only, **before** the configuration is run. That is what makes the
MAE / NPE / R^2 / UCR figures in Table B.11 an honest reliability assessment
rather than an in-sample fit statistic.

Two clarifications the manuscript leaves open
---------------------------------------------
* ``N`` is never stated (amendment A-3). It defaults to
  :data:`DEFAULT_TOTAL_ITERATIONS` = 30, matching Table B.1's 30-row template,
  and random search receives the identical budget so that SER compares like with
  like.
* Random search draws **without replacement** and selects its winner using the
  *same* Pareto + retention + argmax-EIR rule as BOPIS (amendment A-32).
  With replacement, duplicate draws would silently shrink its effective budget
  and the equal-budget premise of SER would fail.

Standard library only.
"""

from __future__ import annotations

import dataclasses
import math
import random
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from bopis import acquisition, tasks
from bopis import config_space as cs
from bopis.config_space import Config
from bopis.gp import GaussianProcess
from bopis.pareto import Evaluation, Objectives, front_mask, pareto_front

#: Total evaluations per method: 10 prior-weighted seeds + 20 BO-guided steps.
DEFAULT_TOTAL_ITERATIONS = 30
DEFAULT_SEED_COUNT = 10

#: Callable that measures one configuration and returns its mean objectives.
EvaluatorFn = Callable[[Config], Objectives]


@dataclasses.dataclass
class IterationRecord:
    """One row of the calibration log (Table B.7)."""

    iteration: int
    config: Config
    source: str  # "seed" | "bo"
    energy_j: float
    tokens_per_s: float
    quality_f1: float
    gp_mu: Optional[float] = None
    gp_sigma: Optional[float] = None
    gp_predictive_sigma: Optional[float] = None
    expected_improvement: Optional[float] = None
    best_energy_so_far: float = math.inf
    delta_energy: float = 0.0
    on_pareto_front: bool = False

    def as_row(self) -> Dict[str, object]:
        row: Dict[str, object] = {
            "iteration": self.iteration,
            "source": self.source,
            "config": self.config.key(),
        }
        row.update(self.config.as_row())
        row.update(
            {
                "energy_j": self.energy_j,
                "tokens_per_s": self.tokens_per_s,
                "quality_f1": self.quality_f1,
                "gp_mu": self.gp_mu,
                "gp_sigma": self.gp_sigma,
                "gp_predictive_sigma": self.gp_predictive_sigma,
                "expected_improvement": self.expected_improvement,
                "best_energy_so_far": self.best_energy_so_far,
                "delta_energy": self.delta_energy,
                "on_pareto_front": self.on_pareto_front,
            }
        )
        return row


@dataclasses.dataclass
class OptimizationResult:
    """Everything one search produced."""

    method: str  # "bopis" | "random_search"
    records: List[IterationRecord]
    evaluations: List[Evaluation]
    gp: Optional[GaussianProcess] = None
    n_seeds: int = 0
    n_total: int = 0
    space_size: int = 0

    # -- derived ------------------------------------------------------------- #

    @property
    def energies(self) -> List[float]:
        return [r.energy_j for r in self.records]

    @property
    def calibration_energy_j(self) -> float:
        """``C_bo``: total energy the search itself consumed."""
        return sum(self.energies)

    def best_record(self) -> Optional[IterationRecord]:
        if not self.records:
            return None
        return min(self.records, key=lambda r: r.energy_j)

    def iteration_of(self, config: Config) -> Optional[int]:
        """1-based iteration at which *config* was first evaluated."""
        for record in self.records:
            if record.config == config:
                return record.iteration
        return None

    def gp_prediction_pairs(
        self,
    ) -> Tuple[List[float], List[float], List[float]]:
        """One-step-ahead ``(predicted, measured, predictive_sigma)`` triples.

        Seed iterations are excluded: no surrogate existed yet, so they carry no
        prediction to score.
        """
        predicted: List[float] = []
        measured: List[float] = []
        sigmas: List[float] = []
        for record in self.records:
            if record.gp_mu is None:
                continue
            predicted.append(record.gp_mu)
            measured.append(record.energy_j)
            sigmas.append(
                record.gp_predictive_sigma
                if record.gp_predictive_sigma is not None
                else float("nan")
            )
        return predicted, measured, sigmas

    def as_dict(self) -> Dict[str, object]:
        return {
            "method": self.method,
            "n_seeds": self.n_seeds,
            "n_total": self.n_total,
            "space_size": self.space_size,
            "calibration_energy_j": self.calibration_energy_j,
            "n_pareto_front": len(pareto_front(self.evaluations)),
            "gp": self.gp.describe() if self.gp else None,
        }


def _finalize(result: OptimizationResult) -> OptimizationResult:
    """Annotate records with best-so-far, delta and front membership."""
    running = math.inf
    for record in result.records:
        previous = running
        running = min(running, record.energy_j)
        record.best_energy_so_far = running
        record.delta_energy = 0.0 if math.isinf(previous) else previous - running

    mask = front_mask(result.evaluations)
    for record, on_front in zip(result.records, mask):
        record.on_pareto_front = on_front
    return result


# --------------------------------------------------------------------------- #
# Bayesian Optimization
# --------------------------------------------------------------------------- #


def bayes_optimize(
    space: Sequence[Config],
    evaluate: EvaluatorFn,
    precision_prior: Mapping[str, float],
    n_total: int = DEFAULT_TOTAL_ITERATIONS,
    n_seeds: int = DEFAULT_SEED_COUNT,
    seed: int = 0,
    total_layers: int = 32,
    ard: bool = False,
    xi: float = 0.0,
    on_iteration: Optional[Callable[[IterationRecord], None]] = None,
) -> OptimizationResult:
    """Run the BOPIS search over *space*.

    Args:
        space: ``X_feasible``.
        evaluate: Measures a configuration on the proxy prompt subset.
        precision_prior: ``P(precision)`` for seeding; masked and renormalized to
            the variants actually present in *space*.
        n_total: ``N`` -- total evaluations, seeds included.
        n_seeds: Size of the initial prior-weighted random phase.
        seed: RNG seed, for reproducibility.
        total_layers: Model depth, for encoding the ``All`` GPU-layer sentinel.
        ard: Per-dimension length scales. Off by default: with at most 30
            observations, seven hyperparameters cannot be identified reliably.
        xi: Exploration margin for Expected Improvement.
        on_iteration: Called after each evaluation, for incremental logging so a
            long run can be resumed after an interruption.
    """
    if not space:
        raise ValueError("cannot optimize over an empty configuration space")
    n_total = min(n_total, len(space))
    n_seeds = min(n_seeds, n_total)

    rng = random.Random(seed)
    encode = lambda cfg: cs.encode(cfg, total_layers=total_layers)  # noqa: E731

    records: List[IterationRecord] = []
    evaluations: List[Evaluation] = []
    evaluated: List[Config] = []
    gp: Optional[GaussianProcess] = None

    # -- Step 1: initialize with prior-weighted seeds ------------------------ #
    seeds = tasks.sample_seed_configs(space, n_seeds, precision_prior, rng)
    for index, config in enumerate(seeds, start=1):
        objectives = evaluate(config)
        record = IterationRecord(
            iteration=index,
            config=config,
            source="seed",
            energy_j=objectives.energy_j,
            tokens_per_s=objectives.tokens_per_s,
            quality_f1=objectives.quality_f1,
        )
        records.append(record)
        evaluations.append(
            Evaluation(config, objectives, iteration=index, source="seed")
        )
        evaluated.append(config)
        if on_iteration:
            on_iteration(record)

    # -- Steps 2-5: fit, acquire, evaluate, update --------------------------- #
    for index in range(n_seeds + 1, n_total + 1):
        x_train = [encode(cfg) for cfg in evaluated]
        y_train = [e.energy_j for e in evaluations]

        # Step 2 / Step 5: (re)fit the GP to every observation so far. The
        # previous fit's hyperparameters warm-start the search, since successive
        # fits differ by exactly one observation.
        previous_hyper = gp.hyper if gp is not None else None
        gp = GaussianProcess(ard=ard, seed=seed + index)
        gp.warm_start = previous_hyper
        gp.fit(x_train, y_train)
        f_best = min(y_train)

        # Step 3: exact argmax of Expected Improvement over the feasible set.
        candidate, best_ei, _all_ei = acquisition.argmax_expected_improvement(
            candidates=space,
            encode=encode,
            predict=gp.predict,
            f_best=f_best,
            xi=xi,
            exclude=evaluated,
        )
        if candidate is None:  # space exhausted
            break

        # Record the prediction BEFORE measuring: this is the one-step-ahead
        # forecast that Table B.11's reliability metrics are computed from.
        mu, sigma = gp.predict(encode(candidate))
        predictive_sigma = gp.predictive_std(encode(candidate))

        # Step 4: evaluate.
        objectives = evaluate(candidate)
        record = IterationRecord(
            iteration=index,
            config=candidate,
            source="bo",
            energy_j=objectives.energy_j,
            tokens_per_s=objectives.tokens_per_s,
            quality_f1=objectives.quality_f1,
            gp_mu=mu,
            gp_sigma=sigma,
            gp_predictive_sigma=predictive_sigma,
            expected_improvement=best_ei,
        )
        records.append(record)
        evaluations.append(
            Evaluation(candidate, objectives, iteration=index, source="bo")
        )
        evaluated.append(candidate)
        if on_iteration:
            on_iteration(record)

    # Refit once more so the returned GP reflects every observation.
    if evaluated:
        gp = GaussianProcess(ard=ard, seed=seed).fit(
            [encode(cfg) for cfg in evaluated],
            [e.energy_j for e in evaluations],
        )

    return _finalize(
        OptimizationResult(
            method="bopis",
            records=records,
            evaluations=evaluations,
            gp=gp,
            n_seeds=n_seeds,
            n_total=n_total,
            space_size=len(space),
        )
    )


# --------------------------------------------------------------------------- #
# Random-search baseline
# --------------------------------------------------------------------------- #


def random_search(
    space: Sequence[Config],
    evaluate: EvaluatorFn,
    n_total: int = DEFAULT_TOTAL_ITERATIONS,
    seed: int = 0,
    on_iteration: Optional[Callable[[IterationRecord], None]] = None,
) -> OptimizationResult:
    """Uniform random search over *space*, without replacement.

    The uninformed baseline of Bergstra and Bengio (2012): it explores the same
    space as BOPIS with the same evaluation budget but no model-guided
    decision-making, which is precisely what isolates the contribution of the
    surrogate.
    """
    if not space:
        raise ValueError("cannot search an empty configuration space")
    n_total = min(n_total, len(space))

    rng = random.Random(seed)
    sampled = rng.sample(list(space), n_total)

    records: List[IterationRecord] = []
    evaluations: List[Evaluation] = []
    for index, config in enumerate(sampled, start=1):
        objectives = evaluate(config)
        record = IterationRecord(
            iteration=index,
            config=config,
            source="random_search",
            energy_j=objectives.energy_j,
            tokens_per_s=objectives.tokens_per_s,
            quality_f1=objectives.quality_f1,
        )
        records.append(record)
        evaluations.append(
            Evaluation(config, objectives, iteration=index, source="random_search")
        )
        if on_iteration:
            on_iteration(record)

    return _finalize(
        OptimizationResult(
            method="random_search",
            records=records,
            evaluations=evaluations,
            gp=None,
            n_seeds=0,
            n_total=n_total,
            space_size=len(space),
        )
    )
