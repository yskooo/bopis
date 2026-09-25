"""Study orchestration: search, three-way validation, statistics, artifacts.

Implements the five procedural stages of Chapter 3's Research Design as a single
callable pipeline, and writes every Appendix B table as it goes.

Stage order
-----------
0. **Reference evaluation.** The unoptimized default configuration is measured on
   the proxy subset *outside both methods' budgets*. Chapter 3 needs ``SRR`` and
   ``QRR`` during ``x*`` selection but defines them only against the 500-prompt
   default, which does not exist yet at selection time (amendment A-33). This
   iteration-0 reference supplies search-time retention ratios; the reported
   ratios are recomputed on the full set in stage 3.
1. **BOPIS search** on the proxy subset -- Chapter 3's six-step BO procedure.
2. **Random search** on the proxy subset, with the identical budget and the
   identical selection rule, so ``SER`` compares like with like.
3. **Three-way validation** on the full evaluation set: unoptimized default,
   the random-search winner, and ``x*``.
4. **Analysis** -- descriptives, Friedman + Nemenyi per dependent variable,
   success indicators, surrogate reliability, hypervolume, amortization.

Standard library only.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from typing import Callable, Dict, List, Optional, Sequence

from bopis import artifacts, metrics, optimizer, schemas, stats, tasks
from bopis import config_space as cs
from bopis.config_space import Config
from bopis.dataset import Prompt, SampleSet
from bopis.hardware import HostProfile
from bopis.measure import Measurer, PromptMeasurement, aggregate
from bopis.monitor import estimator
from bopis.pareto import (
    Objectives,
    Selection,
    hypervolume,
    pareto_front,
    select_xstar,
)


@dataclasses.dataclass
class RunSettings:
    """Everything that controls a run, recorded verbatim in the manifest."""

    backend: str = "sim"
    n_total: int = optimizer.DEFAULT_TOTAL_ITERATIONS
    n_seeds: int = optimizer.DEFAULT_SEED_COUNT
    seed: int = 0
    eval_size: int = 500
    proxy_size: int = 50
    ard: bool = False
    xi: float = 0.0
    min_gpu_layers: Optional[int] = None
    allow_no_power: bool = False

    #: ``auto`` uses the NVML ladder and refuses when it has no instrument.
    #: ``resource-estimate`` adds the labelled fallback in
    #: :mod:`bopis.monitor.estimator` below that ladder.
    energy_mode: str = "auto"
    estimate_cpu_tdp_w: float = estimator.DEFAULT_CPU_TDP_W
    estimate_gpu_tdp_w: float = estimator.DEFAULT_GPU_TDP_W
    estimate_cpu_idle_w: float = 0.0
    estimate_gpu_idle_w: float = 0.0
    estimate_uncertainty: float = estimator.DEFAULT_UNCERTAINTY_FRAC

    tariff_php_per_kwh: float = metrics.DEFAULT_TARIFF_PHP_PER_KWH
    ctx_size: int = cs.FIXED_CTX_SIZE
    total_layers: int = 32
    resume: bool = False
    skip_validation: bool = False
    #: Overrides the default's ``g`` (Table 3.2 says "All"); set to 0 when the
    #: energy instrument covers only the CPU package.
    default_gpu_layers: Optional[int] = None
    #: ``cei`` -- Expected Improvement weighted by the probability of meeting
    #: the QRR/SRR floors (A-42); ``ei`` -- Chapter 3's energy-only EI.
    acquisition: str = "cei"

    #: Which instrument measured energy, when it is not implied by the backend
    #: (e.g. the RAPL feed's URL, sensor path and measured refresh interval).
    energy_instrument: Optional[Dict[str, object]] = None
    #: The quality scorer's own description, or None when quality is unscored.
    quality_scorer: Optional[Dict[str, object]] = None

    @property
    def estimates_energy(self) -> bool:
        return self.energy_mode == "resource-estimate"

    def power_budget(self) -> estimator.PowerBudget:
        """The declared power envelope, for the estimator to scale."""
        return estimator.PowerBudget(
            cpu_tdp_w=self.estimate_cpu_tdp_w,
            gpu_tdp_w=self.estimate_gpu_tdp_w,
            cpu_idle_w=self.estimate_cpu_idle_w,
            gpu_idle_w=self.estimate_gpu_idle_w,
            uncertainty_frac=self.estimate_uncertainty,
        )

    def as_dict(self) -> Dict[str, object]:
        payload = dataclasses.asdict(self)
        payload["t_semantics"] = (
            "t is the maximum generation length (n_predict); --ctx-size is "
            "fixed outside the search space (amendment A-1)"
        )
        # The estimator's formula, inputs, assumptions and caveat go in the
        # manifest verbatim. A reader who disagrees with an assumption can then
        # see which one, without needing the source that produced the run.
        if self.estimates_energy:
            payload["energy_estimator"] = estimator.describe(
                self.power_budget()
            )
        return payload


@dataclasses.dataclass
class StudyResult:
    """Everything a completed run produced."""

    run_dir: artifacts.RunDirectory
    settings: RunSettings
    profile: HostProfile
    space: List[Config]
    bo: optimizer.OptimizationResult
    rs: optimizer.OptimizationResult
    bo_selection: Selection
    rs_selection: Selection
    search_reference: Objectives
    validation: Dict[str, List[PromptMeasurement]]
    summary: Dict[str, object]

    @property
    def xstar(self) -> Optional[Config]:
        return self.bo_selection.config


ProgressFn = Callable[[str], None]


def _noop(_message: str) -> None:
    pass


class Study:
    """One end-to-end BOPIS study."""

    def __init__(
        self,
        run_dir: artifacts.RunDirectory,
        settings: RunSettings,
        profile: HostProfile,
        space: Sequence[Config],
        samples: SampleSet,
        measurer: Measurer,
        backend_start: Optional[Callable[[Config], None]] = None,
        backend_stop: Optional[Callable[[], None]] = None,
        progress: ProgressFn = _noop,
        scorer=None,
    ) -> None:
        self.run_dir = run_dir
        #: A :class:`bopis.quality.Scorer`, or None. Only the simulator fills
        #: ``quality_f1`` itself; for real backends this is what does.
        self.scorer = scorer
        self.settings = settings
        self.profile = profile
        self.space = list(space)
        self.samples = samples
        self.measurer = measurer
        self.backend_start = backend_start
        self.backend_stop = backend_stop
        self.progress = progress

        self._proxy_cache: Dict[str, Objectives] = {}
        self._calibration_writer: Optional[artifacts.TableWriter] = None

    # ------------------------------------------------------------------ #
    # Evaluation on the proxy subset (used during search)
    # ------------------------------------------------------------------ #

    def _measure_set(
        self, config: Config, prompts: Sequence[Prompt], condition: str
    ) -> List[PromptMeasurement]:
        """Measure *config* over *prompts*, restarting the backend once."""
        if self.backend_start:
            self.backend_start(config)
        try:
            results: List[PromptMeasurement] = []
            for prompt in prompts:
                results.append(
                    self.measurer.measure(
                        prompt_index=prompt.index,
                        prompt_id=prompt.prompt_id,
                        task_type=prompt.task_type,
                        condition=condition,
                        config=config,
                        prompt=prompt.text,
                    )
                )
        finally:
            if self.backend_stop:
                self.backend_stop()
        # Scored only after the server has been stopped: every energy window
        # of this batch is closed, and the transformer forward pass cannot
        # compete with inference for the CPU it is being measured on.
        self._score_quality(results, prompts)
        self._write_generations(results, prompts)
        return results

    def _score_quality(
        self, results: List[PromptMeasurement], prompts: Sequence[Prompt]
    ) -> None:
        """BERTScore every unscored generation against its Dolly reference."""
        if self.scorer is None:
            return
        pending = [
            (m, p)
            for m, p in zip(results, prompts)
            if m.quality_f1 is None and m.error is None
        ]
        # An empty answer is a real, bad outcome, not a missing one; dropping
        # it would raise the mean. Scored as 0, the rescaled random-pair level.
        for m, _p in pending:
            if not m.text.strip():
                m.quality_f1 = m.quality_precision = m.quality_recall = 0.0
                m.quality_scorer = f"{self.scorer.name}:empty_candidate"
        pending = [(m, p) for m, p in pending if m.quality_f1 is None]
        if not pending:
            return
        scores = self.scorer.score(
            [m.text for m, _p in pending], [p.response for _m, p in pending]
        )
        for (m, _p), score in zip(pending, scores):
            m.quality_f1 = score.f1
            m.quality_precision = score.precision
            m.quality_recall = score.recall
            m.quality_scorer = score.scorer
            m.baseline_rescaled = score.baseline_rescaled

    def _write_generations(
        self, results: List[PromptMeasurement], prompts: Sequence[Prompt]
    ) -> None:
        """Persist generated text, so quality can be audited or re-scored."""
        if not results or not any(m.text for m in results):
            return
        path = self.run_dir.path("raw", "generations.jsonl")
        with open(path, "a", encoding="utf-8") as handle:
            for m, p in zip(results, prompts):
                handle.write(
                    json.dumps(
                        {
                            "condition": m.condition,
                            "config": m.config.key(),
                            "prompt_id": m.prompt_id,
                            "task_type": m.task_type,
                            "candidate": m.text,
                            "reference": p.response,
                            "quality_f1": m.quality_f1,
                            "scorer": m.quality_scorer,
                            "energy_j": m.energy_j,
                            "energy_method": m.energy_method,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def evaluate_on_proxy(self, config: Config) -> Objectives:
        """Mean objectives for *config* over the 50-prompt proxy subset.

        Memoized, because random search and BOPIS can propose the same
        configuration and re-measuring it would waste a slot of the budget while
        adding nothing.
        """
        key = config.key()
        cached = self._proxy_cache.get(key)
        if cached is not None:
            return cached

        measurements = self._measure_set(config, self.samples.proxy, "search")
        energies = aggregate(measurements, "energy_j")
        speeds = aggregate(measurements, "tokens_per_s")
        qualities = aggregate(measurements, "quality_f1")

        objectives = Objectives(
            energy_j=sum(energies) / len(energies) if energies else math.inf,
            tokens_per_s=sum(speeds) / len(speeds) if speeds else 0.0,
            quality_f1=sum(qualities) / len(qualities) if qualities else 0.0,
        )
        self._proxy_cache[key] = objectives
        return objectives

    def reference_default(self) -> Config:
        """The unoptimized default this study is measured against.

        Table 3.2's default, with ``g`` overridden when the energy instrument
        requires it. On a real backend the default must also be *measurable*:
        if it falls outside ``X_feasible`` -- typically because its GGUF (the
        default model at F16) was not supplied -- the closest feasible
        configuration with the same runtime settings stands in: same model if
        possible, otherwise the nearest rung, at the highest precision
        available. The simulator can evaluate anything, so it keeps the
        literal default and stays comparable with earlier runs.
        """
        default = cs.default_config(self.profile.physical_cores)
        if self.settings.default_gpu_layers is not None:
            default = default._replace(g=self.settings.default_gpu_layers)
        if self.settings.backend == "sim" or default in self.space or not self.space:
            return default

        same_runtime = [
            cfg
            for cfg in self.space
            if (cfg.t, cfg.b, cfg.g, cfg.c) == (default.t, default.b, default.g, default.c)
        ] or list(self.space)
        default_params = cs.MODELS[cs.DEFAULT_M].n_params

        def distance(cfg: Config) -> tuple:
            variant = cs.MODELS.get(cfg.m)
            params = variant.n_params if variant else default_params
            return (
                cfg.m != default.m,
                abs(math.log(params / default_params)),
                cs.P_VALUES.index(cfg.p) if cfg.p in cs.P_VALUES else 99,
                abs(cfg.t - default.t),
                -cfg.c,
            )

        chosen = min(same_runtime, key=distance)
        self.progress(
            f"        default {default} is not measurable here (not in X_feasible); "
            f"nearest feasible stand-in: {chosen}"
        )
        self.default_substituted_for = default
        return chosen

    #: Set by :meth:`reference_default` when the literal default was replaced.
    default_substituted_for: Optional[Config] = None

    def _search_floors(self, reference: Objectives) -> Optional[optimizer.Floors]:
        """Absolute QRR/SRR floors for constrained acquisition, or None.

        The same thresholds :func:`select_xstar` applies afterwards, so the
        search looks where the selection rule will accept an answer. A floor
        whose reference value is missing -- unscored quality on a real run --
        is dropped rather than set to zero, which would make it vacuous.
        """
        if self.settings.acquisition != "cei":
            return None

        def usable(value: float) -> bool:
            return value is not None and math.isfinite(value) and value > 0

        return optimizer.Floors(
            quality_f1=(
                reference.quality_f1 * metrics.QRR_THRESHOLD / 100.0
                if usable(reference.quality_f1)
                else None
            ),
            tokens_per_s=(
                reference.tokens_per_s * metrics.SRR_THRESHOLD / 100.0
                if usable(reference.tokens_per_s)
                else None
            ),
        )

    # ------------------------------------------------------------------ #
    # Stages
    # ------------------------------------------------------------------ #

    def run(self) -> StudyResult:
        started = time.perf_counter()

        # -- Stage 0: iteration-0 reference ---------------------------- #
        default_config = self.reference_default()
        self.progress(f"Stage 0: reference evaluation of default {default_config}")
        search_reference = self.evaluate_on_proxy(default_config)

        # -- Stage 1: BOPIS search ------------------------------------- #
        prior = tasks.dataset_prior(self.samples.task_proportions())
        self._task_prior = prior
        self.progress(
            f"Stage 1: BOPIS search, {self.settings.n_total} evaluations "
            f"({self.settings.n_seeds} prior-weighted seeds) over "
            f"{len(self.space)} feasible configurations"
        )
        with self.run_dir.writer("B.7") as writer:
            bo = optimizer.bayes_optimize(
                space=self.space,
                evaluate=self.evaluate_on_proxy,
                precision_prior=prior,
                n_total=self.settings.n_total,
                n_seeds=self.settings.n_seeds,
                seed=self.settings.seed,
                total_layers=self.settings.total_layers,
                ard=self.settings.ard,
                xi=self.settings.xi,
                on_iteration=lambda record: writer.write(record.as_row()),
                floors=self._search_floors(search_reference),
            )

        # -- Stage 2: random-search baseline --------------------------- #
        self.progress(
            f"Stage 2: random-search baseline, matched budget of "
            f"{self.settings.n_total} evaluations"
        )
        rs = optimizer.random_search(
            space=self.space,
            evaluate=self.evaluate_on_proxy,
            n_total=self.settings.n_total,
            seed=self.settings.seed + 9973,
        )
        self.run_dir.write_table(
            "B.7", [r.as_row() for r in bo.records]
        )  # rewrite with front flags now known
        rs_log = self.run_dir.path("calibration", "rs_log.csv")
        with artifacts.TableWriter(rs_log, "B.7") as writer:
            writer.write_all(r.as_row() for r in rs.records)

        # -- Selection -------------------------------------------------- #
        bo_selection = select_xstar(bo.evaluations, search_reference)
        rs_selection = select_xstar(rs.evaluations, search_reference)
        self.progress(
            f"        x* = {bo_selection.config} "
            f"[{bo_selection.status}]  |  random-search pick = "
            f"{rs_selection.config}"
        )

        # -- Stage 3: three-way validation ------------------------------ #
        validation: Dict[str, List[PromptMeasurement]] = {}
        if self.settings.skip_validation:
            self.progress("Stage 3: skipped (--skip-validation)")
        else:
            condition_configs = [
                ("unoptimized", default_config),
                ("random_search", rs_selection.config or default_config),
                ("bopis", bo_selection.config or default_config),
            ]
            for condition, config in condition_configs:
                self.progress(
                    f"Stage 3: validating '{condition}' on "
                    f"{len(self.samples.evaluation)} prompts with {config}"
                )
                validation[condition] = self._measure_set(
                    config, self.samples.evaluation, condition
                )

        # -- Stage 4: analysis ------------------------------------------ #
        self.progress("Stage 4: analysis and artifact generation")
        summary = self._analyze(
            default_config=default_config,
            bo=bo,
            rs=rs,
            bo_selection=bo_selection,
            rs_selection=rs_selection,
            search_reference=search_reference,
            validation=validation,
        )
        summary["wall_seconds"] = round(time.perf_counter() - started, 2)
        # The seeding prior, so the dashboard can show what weighted the seeds.
        summary["task_prior"] = {
            "dataset_prior": dict(self._task_prior),
            "task_proportions": dict(self.samples.task_proportions()),
        }
        # A substituted reference changes what EIR/SRR/QRR are relative to, so
        # it travels with the results rather than living only in the log.
        summary["default_config"] = default_config.key()
        summary["default_substituted_for"] = (
            self.default_substituted_for.key()
            if self.default_substituted_for is not None
            else None
        )

        return StudyResult(
            run_dir=self.run_dir,
            settings=self.settings,
            profile=self.profile,
            space=self.space,
            bo=bo,
            rs=rs,
            bo_selection=bo_selection,
            rs_selection=rs_selection,
            search_reference=search_reference,
            validation=validation,
            summary=summary,
        )

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #

    def _condition_objectives(
        self, measurements: Sequence[PromptMeasurement]
    ) -> Objectives:
        energies = aggregate(list(measurements), "energy_j")
        speeds = aggregate(list(measurements), "tokens_per_s")
        qualities = aggregate(list(measurements), "quality_f1")
        return Objectives(
            energy_j=sum(energies) / len(energies) if energies else math.nan,
            tokens_per_s=sum(speeds) / len(speeds) if speeds else math.nan,
            quality_f1=sum(qualities) / len(qualities) if qualities else math.nan,
        )

    def _analyze(
        self,
        default_config: Config,
        bo: optimizer.OptimizationResult,
        rs: optimizer.OptimizationResult,
        bo_selection: Selection,
        rs_selection: Selection,
        search_reference: Objectives,
        validation: Dict[str, List[PromptMeasurement]],
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {}

        # -- Surrogate reliability (Table B.11) ------------------------- #
        predicted, measured, sigmas = bo.gp_prediction_pairs()
        reliability = metrics.surrogate_reliability(
            predicted,
            measured,
            sigmas,
            basis=metrics.ReliabilityBasis.ONE_STEP_AHEAD,
        )
        payload["surrogate_reliability"] = reliability.as_dict()

        surrogate_rows = []
        for record in bo.records:
            if record.gp_mu is None:
                continue
            error = abs(record.gp_mu - record.energy_j)
            sigma = record.gp_predictive_sigma
            surrogate_rows.append(
                {
                    "iteration": record.iteration,
                    "config": record.config.key(),
                    "gp_mu": record.gp_mu,
                    "gp_sigma": record.gp_sigma,
                    "gp_predictive_sigma": sigma,
                    "energy_measured_j": record.energy_j,
                    "abs_error": error,
                    "within_95_ci": (
                        None if sigma is None else error <= 1.96 * sigma
                    ),
                }
            )
        self.run_dir.write_table("B.11", surrogate_rows)

        # Leave-one-out reliability -- a far more honest estimate at n<=30 than
        # the one-step-ahead sequence alone (amendment A-36).
        if bo.gp is not None:
            loo = bo.gp.leave_one_out()
            if loo:
                payload["surrogate_loo"] = metrics.surrogate_reliability(
                    [mean for _actual, mean, _sd in loo],
                    [actual for actual, _mean, _sd in loo],
                    [sd for _a, _m, sd in loo],
                    basis=metrics.ReliabilityBasis.LEAVE_ONE_OUT,
                ).as_dict()
            payload["gp"] = bo.gp.describe()

        # -- Convergence and sample efficiency -------------------------- #
        k_bo = bo.iteration_of(bo_selection.config) if bo_selection.config else None
        k_rs = rs.iteration_of(rs_selection.config) if rs_selection.config else None
        ser = (
            metrics.sample_efficiency_ratio(k_rs, k_bo)
            if (k_bo and k_rs)
            else None
        )
        payload["convergence"] = {
            "bopis": {
                "energies": bo.energies,
                "best_so_far": metrics.best_so_far(bo.energies),
                "delta_energy": metrics.improvement_per_iteration(bo.energies),
                "iterations_to_within_5pct": metrics.iterations_to_within(bo.energies),
                "k_star": k_bo,
            },
            "random_search": {
                "energies": rs.energies,
                "best_so_far": metrics.best_so_far(rs.energies),
                "delta_energy": metrics.improvement_per_iteration(rs.energies),
                "iterations_to_within_5pct": metrics.iterations_to_within(rs.energies),
                "k_star": k_rs,
            },
            "ser": ser,
            "ser_note": (
                "Single-run point estimate; k* is an iteration index, so this is "
                "one draw from a high-variance statistic. Report a mean over "
                "seeded repeats where possible (amendment A-16)."
            ),
        }

        # -- Pareto front and hypervolume (Table B.10) ------------------ #
        front = pareto_front(bo.evaluations)
        front_ids = {id(e) for e in front}
        hv = hypervolume(front, search_reference, normalize=True)
        payload["pareto"] = {
            "n_evaluated": len(bo.evaluations),
            "n_front": len(front),
            "hypervolume_normalized": hv,
            "reference_point": {
                "energy_j": search_reference.energy_j,
                "tokens_per_s": search_reference.tokens_per_s,
                "quality_f1": search_reference.quality_f1,
            },
            "hypervolume_note": (
                "Objectives are min-max normalized against the reference point "
                "before integration, so HV is in [0,1] and comparable across "
                "machines (amendment A-15)."
            ),
        }
        self.run_dir.write_table(
            "B.10",
            [
                {
                    "config": e.config.key(),
                    "method": e.source,
                    "iteration": e.iteration,
                    **e.config.as_row(),
                    "energy_j": e.energy_j,
                    "tokens_per_s": e.tokens_per_s,
                    "quality_f1": e.quality_f1,
                    "on_pareto_front": id(e) in front_ids,
                    "is_xstar": e.config == bo_selection.config,
                }
                for e in bo.evaluations
            ],
        )

        # -- Selection table (Table B.9) -------------------------------- #
        ranked = sorted(front, key=lambda e: e.energy_j)
        selection_rows = []
        for rank, evaluation in enumerate(ranked, start=1):
            eir = metrics.energy_improvement_ratio(
                search_reference.energy_j, evaluation.energy_j
            )
            srr = metrics.speed_retention_ratio(
                search_reference.tokens_per_s, evaluation.tokens_per_s
            )
            qrr = metrics.quality_retention_ratio(
                search_reference.quality_f1, evaluation.quality_f1
            )
            selection_rows.append(
                {
                    "rank": rank,
                    "config": evaluation.config.key(),
                    **evaluation.config.as_row(),
                    "energy_j": evaluation.energy_j,
                    "tokens_per_s": evaluation.tokens_per_s,
                    "quality_f1": evaluation.quality_f1,
                    "eir_percent": eir,
                    "srr_percent": srr,
                    "qrr_percent": qrr,
                    "meets_srr": srr >= metrics.SRR_THRESHOLD,
                    "meets_qrr": qrr >= metrics.QRR_THRESHOLD,
                    "on_pareto_front": True,
                    "selected": evaluation.config == bo_selection.config,
                    "selection_status": bo_selection.status,
                }
            )
        self.run_dir.write_table("B.9", selection_rows)
        payload["selection"] = bo_selection.as_dict()
        payload["random_search_selection"] = rs_selection.as_dict()
        payload["search_reference"] = {
            "config": default_config.key(),
            "energy_j": search_reference.energy_j,
            "tokens_per_s": search_reference.tokens_per_s,
            "quality_f1": search_reference.quality_f1,
            "note": (
                "Iteration-0 reference on the proxy subset, outside both "
                "methods' budgets, used for search-time SRR/QRR "
                "(amendment A-33)."
            ),
        }

        # -- Configuration evaluation table (Table B.2) ----------------- #
        config_rows = []
        for result in (bo, rs):
            for record in result.records:
                config_rows.append(
                    {
                        "config_id": record.config.key(),
                        "method": result.method,
                        "iteration": record.iteration,
                        **record.config.as_row(),
                        "energy_j": record.energy_j,
                        "tokens_per_s": record.tokens_per_s,
                        "quality_f1": record.quality_f1,
                        "on_pareto_front": record.on_pareto_front,
                    }
                )
        self.run_dir.write_table("B.2", config_rows)

        # -- Calibration cost ------------------------------------------- #
        # C_bo: the energy the search itself spent, which the amortization
        # analysis needs to know before it can compute a break-even point.
        self.summary_calibration_energy = bo.calibration_energy_j
        payload["calibration"] = {
            "bopis_energy_j": bo.calibration_energy_j,
            "random_search_energy_j": rs.calibration_energy_j,
            "n_evaluations_bopis": len(bo.records),
            "n_evaluations_random_search": len(rs.records),
            "n_unique_configs_measured": len(self._proxy_cache),
        }

        if not validation:
            payload["validation"] = None
            self.run_dir.write_metrics(payload)
            return payload

        payload.update(self._analyze_validation(validation, default_config))
        self.run_dir.write_metrics(payload)
        return payload

    def _analyze_validation(
        self,
        validation: Dict[str, List[PromptMeasurement]],
        default_config: Config,
    ) -> Dict[str, object]:
        """Per-prompt tables, descriptives, Friedman/Nemenyi, indicators."""
        payload: Dict[str, object] = {}

        default_objectives = self._condition_objectives(
            validation.get("unoptimized", [])
        )
        s_min = metrics.s_min(default_objectives.tokens_per_s)

        # Per-task Q_min, since Table B.5's threshold is per task category.
        per_task_default: Dict[str, List[float]] = {}
        for m in validation.get("unoptimized", []):
            if m.quality_f1 is not None:
                per_task_default.setdefault(m.task_type, []).append(m.quality_f1)
        q_min_by_task = {
            task: metrics.q_min(sum(values) / len(values))
            for task, values in per_task_default.items()
            if values
        }

        # -- Per-prompt tables B.3-B.6 ---------------------------------- #
        writers = {
            "B.3": self.run_dir.writer("B.3"),
            "B.4": self.run_dir.writer("B.4"),
            "B.5": self.run_dir.writer("B.5"),
            "B.6": self.run_dir.writer("B.6"),
        }
        try:
            for condition in schemas.CONDITIONS:
                for m in validation.get(condition, []):
                    writers["B.3"].write(m.energy_row())
                    writers["B.4"].write(m.speed_row(s_min))
                    writers["B.5"].write(
                        m.quality_row(q_min_by_task.get(m.task_type))
                    )
                    writers["B.6"].write(m.resource_row())
        finally:
            for writer in writers.values():
                writer.close()

        # -- Descriptives and the Friedman/Nemenyi block (Table B.8) ---- #
        descriptives: Dict[str, Dict[str, object]] = {}
        tests: Dict[str, object] = {}
        summary_rows: List[Dict[str, object]] = []

        for column, label, higher_is_better in schemas.DEPENDENT_VARIABLES:
            per_condition = {
                condition: aggregate(validation.get(condition, []), column)
                for condition in schemas.CONDITIONS
            }
            descriptives[column] = {
                condition: stats.describe(values).as_dict()
                for condition, values in per_condition.items()
            }
            for statistic in ("mean", "sd", "min", "max"):
                row: Dict[str, object] = {
                    "dependent_variable": column,
                    "dv_label": label,
                    "statistic": statistic,
                }
                for condition in schemas.CONDITIONS:
                    row[condition] = descriptives[column][condition][statistic]
                summary_rows.append(row)

            # The Friedman test needs complete blocks: every condition must have
            # a value for the same prompt.
            lengths = {len(v) for v in per_condition.values()}
            if len(lengths) == 1 and lengths != {0}:
                blocks = list(
                    zip(*(per_condition[c] for c in schemas.CONDITIONS))
                )
                tests[column] = stats.compare_conditions(
                    blocks,
                    condition_names=list(schemas.CONDITIONS),
                    higher_is_better=higher_is_better,
                )
                friedman = tests[column]["friedman"]  # type: ignore[index]
                summary_rows.append(
                    {
                        "dependent_variable": column,
                        "dv_label": label,
                        "statistic": "friedman_chi2",
                        **{c: friedman["chi_square"] for c in schemas.CONDITIONS},
                    }
                )
                summary_rows.append(
                    {
                        "dependent_variable": column,
                        "dv_label": label,
                        "statistic": "friedman_p",
                        **{c: friedman["p_value"] for c in schemas.CONDITIONS},
                    }
                )
            else:
                tests[column] = {
                    "friedman": None,
                    "nemenyi": None,
                    "skipped": "incomplete blocks across conditions",
                }

        self.run_dir.write_table("B.8", summary_rows)
        payload["descriptives"] = descriptives
        payload["statistical_tests"] = tests

        # -- Success indicators ----------------------------------------- #
        bopis_objectives = self._condition_objectives(validation.get("bopis", []))
        rs_objectives = self._condition_objectives(
            validation.get("random_search", [])
        )
        indicators = metrics.success_indicators(
            energy_default=default_objectives.energy_j,
            energy_optimized=bopis_objectives.energy_j,
            speed_default=default_objectives.tokens_per_s,
            speed_optimized=bopis_objectives.tokens_per_s,
            quality_default=default_objectives.quality_f1,
            quality_optimized=bopis_objectives.quality_f1,
        )
        payload["success_indicators"] = indicators.as_dict()
        payload["random_search_indicators"] = metrics.success_indicators(
            energy_default=default_objectives.energy_j,
            energy_optimized=rs_objectives.energy_j,
            speed_default=default_objectives.tokens_per_s,
            speed_optimized=rs_objectives.tokens_per_s,
            quality_default=default_objectives.quality_f1,
            quality_optimized=rs_objectives.quality_f1,
        ).as_dict()
        payload["s_min"] = s_min
        payload["q_min_by_task"] = q_min_by_task

        # -- Per-variant descriptive comparison ------------------------- #
        # Chapter 3 asks for a descriptive comparison across F32/F16/Q8_0/
        # Q4_K_M. Note that in the three-way design only the variants actually
        # chosen by a condition appear here, so this is not a controlled
        # per-variant experiment -- run `bopis sweep` for that.
        variant_energy: Dict[str, List[float]] = {}
        variant_speed: Dict[str, List[float]] = {}
        variant_quality: Dict[str, List[float]] = {}
        variant_conditions: Dict[str, set] = {}

        for condition, measurements in validation.items():
            for m in measurements:
                variant = m.config.p
                variant_conditions.setdefault(variant, set()).add(condition)
                if m.energy_j is not None:
                    variant_energy.setdefault(variant, []).append(m.energy_j)
                variant_speed.setdefault(variant, []).append(m.tokens_per_s)
                if m.quality_f1 is not None:
                    variant_quality.setdefault(variant, []).append(m.quality_f1)

        payload["per_variant"] = {
            variant: {
                "conditions": sorted(conditions),
                "energy_j": stats.describe(
                    variant_energy.get(variant, [])
                ).as_dict(),
                "tokens_per_s": stats.describe(
                    variant_speed.get(variant, [])
                ).as_dict(),
                "quality_f1": stats.describe(
                    variant_quality.get(variant, [])
                ).as_dict(),
            }
            for variant, conditions in variant_conditions.items()
        }

        # -- Amortization ------------------------------------------------ #
        amortization = metrics.amortization(
            calibration_energy_j=self.summary_calibration_energy,
            energy_default_per_prompt_j=default_objectives.energy_j,
            energy_optimized_per_prompt_j=bopis_objectives.energy_j,
            tariff_php_per_kwh=self.settings.tariff_php_per_kwh,
        )
        payload["amortization"] = amortization.as_dict()

        # -- Trials table (Table B.1) ------------------------------------ #
        trial_rows = []
        configs = {
            "unoptimized": default_config,
            "random_search": (
                validation["random_search"][0].config
                if validation.get("random_search")
                else default_config
            ),
            "bopis": (
                validation["bopis"][0].config
                if validation.get("bopis")
                else default_config
            ),
        }
        for number, condition in enumerate(schemas.CONDITIONS, start=1):
            measurements = validation.get(condition, [])
            if not measurements:
                continue
            energy = stats.describe(aggregate(measurements, "energy_j"))
            trial_rows.append(
                {
                    "trial_no": number,
                    "condition": condition,
                    "condition_label": schemas.CONDITION_LABELS[condition],
                    "n_prompts": len(measurements),
                    **configs[condition].as_row(),
                    "energy_j_mean": energy.mean,
                    "energy_j_sd": energy.sd,
                    "j_per_token_mean": stats.describe(
                        aggregate(measurements, "j_per_token")
                    ).mean,
                    "tokens_per_s_mean": stats.describe(
                        aggregate(measurements, "tokens_per_s")
                    ).mean,
                    "quality_f1_mean": stats.describe(
                        aggregate(measurements, "quality_f1")
                    ).mean,
                    "cpu_percent_mean": stats.describe(
                        aggregate(measurements, "cpu_percent")
                    ).mean,
                    "gpu_percent_mean": stats.describe(
                        aggregate(measurements, "gpu_percent")
                    ).mean,
                    "memory_mib_mean": stats.describe(
                        aggregate(measurements, "memory_mib")
                    ).mean,
                    "energy_method": measurements[0].energy_method,
                    "energy_scope": measurements[0].energy_scope,
                }
            )
        self.run_dir.write_table("B.1", trial_rows)
        payload["trials"] = trial_rows

        # -- Scope validity warning -------------------------------------- #
        invalid = [
            condition
            for condition, measurements in validation.items()
            if measurements and not measurements[0].scope_valid
        ]
        if invalid:
            payload["scope_warning"] = (
                "Conditions "
                + ", ".join(sorted(invalid))
                + " have energy_scope that does not cover the work performed "
                "(typically gpu_only accounting at g=0). Their energy figures "
                "are not comparable (amendment A-19)."
            )

        # -- Estimated-energy caveat -------------------------------------- #
        # Attached to the analysis payload, not only to the manifest, so that
        # every consumer of this run -- dashboard, tables, thesis text -- has
        # the qualification in hand rather than having to remember it.
        if self.settings.estimates_energy:
            payload["energy_caveat"] = estimator.caveat()
            payload["energy_estimator"] = estimator.describe(
                self.settings.power_budget()
            )

        return payload

    #: Set by :meth:`_analyze` before validation analysis runs.
    summary_calibration_energy: float = 0.0
