"""Builds the payload the dashboard renders.

The dashboard must display *only* values traceable to a run's artifacts. The
mockup it replaces hardcoded every number -- its Pareto front was a literal
array with membership asserted rather than computed -- so the rule here is that
this module is the single bridge between measured data and pixels, and it
derives everything from :class:`bopis.runner.StudyResult`.

The payload is written as ``dashboard_data.js`` assigning ``window.BOPIS_DATA``,
because the dashboard opens from ``file://`` and browsers refuse to ``fetch``
local files.

Standard library only.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from bopis import __version__, metrics, schemas
from bopis.pareto import pareto_front
from bopis.runner import StudyResult


def _config_payload(config) -> Optional[Dict[str, object]]:
    if config is None:
        return None
    return {
        "key": config.key(),
        "label": str(config),
        "t_max_gen_tokens": config.t,
        "b_batch_size": config.b,
        "p_precision": config.p,
        "g_gpu_layers": config.g_label,
        "c_cpu_threads": config.c,
    }


def build_payload(result: StudyResult) -> Dict[str, object]:
    """Assemble everything the dashboard panels need."""
    summary = result.summary
    front = pareto_front(result.bo.evaluations)
    front_ids = {id(e) for e in front}

    # -- Pareto scatter: every evaluated point, flagged --------------------- #
    pareto_points: List[Dict[str, object]] = []
    for evaluation in result.bo.evaluations:
        pareto_points.append(
            {
                "config": evaluation.config.key(),
                "label": str(evaluation.config),
                "precision": evaluation.config.p,
                "iteration": evaluation.iteration,
                "source": evaluation.source,
                "energy_j": evaluation.energy_j,
                "tokens_per_s": evaluation.tokens_per_s,
                "quality_f1": evaluation.quality_f1,
                "on_front": id(evaluation) in front_ids,
                "is_xstar": evaluation.config == result.bo_selection.config,
            }
        )

    random_points = [
        {
            "config": evaluation.config.key(),
            "label": str(evaluation.config),
            "precision": evaluation.config.p,
            "iteration": evaluation.iteration,
            "energy_j": evaluation.energy_j,
            "tokens_per_s": evaluation.tokens_per_s,
            "quality_f1": evaluation.quality_f1,
        }
        for evaluation in result.rs.evaluations
    ]

    # -- GP reliability scatter --------------------------------------------- #
    gp_points = [
        {
            "iteration": record.iteration,
            "config": record.config.key(),
            "mu": record.gp_mu,
            "sigma": record.gp_sigma,
            "predictive_sigma": record.gp_predictive_sigma,
            "measured": record.energy_j,
            "expected_improvement": record.expected_improvement,
        }
        for record in result.bo.records
        if record.gp_mu is not None
    ]

    reference = result.search_reference
    indicators = summary.get("success_indicators")

    payload: Dict[str, object] = {
        "meta": {
            "bopis_version": __version__,
            "run": result.run_dir.root.replace("\\", "/").split("/")[-1],
            "backend": result.settings.backend,
            "seed": result.settings.seed,
            "n_total_iterations": result.settings.n_total,
            "n_seeds": result.settings.n_seeds,
            "wall_seconds": summary.get("wall_seconds"),
            "validated": bool(result.validation),
            # Displayed prominently: a simulated energy figure must never be
            # mistaken for a measured one.
            "energy_method": (
                result.validation["bopis"][0].energy_method
                if result.validation.get("bopis")
                else "n/a"
            ),
            "energy_scope": (
                result.validation["bopis"][0].energy_scope
                if result.validation.get("bopis")
                else "n/a"
            ),
        },
        "host": result.profile.as_dict(),
        "space": {
            "n_feasible": len(result.space),
            "n_unconstrained": 768,
            "rules_fired": result.profile.rules_fired,
            "permitted_precisions": list(result.profile.permitted_precisions),
            "permitted_gpu_layers": [
                "All" if v == -1 else v
                for v in result.profile.permitted_gpu_layers
            ],
            "permitted_batch_sizes": list(result.profile.permitted_batch_sizes),
            "permitted_cpu_threads": list(result.profile.permitted_cpu_threads),
        },
        "selection": result.bo_selection.as_dict(),
        "random_search_selection": result.rs_selection.as_dict(),
        "xstar_config": _config_payload(result.bo_selection.config),
        "default_config": _config_payload(
            result.validation["unoptimized"][0].config
            if result.validation.get("unoptimized")
            else None
        ),
        "reference_point": {
            "energy_j": reference.energy_j,
            "tokens_per_s": reference.tokens_per_s,
            "quality_f1": reference.quality_f1,
        },
        "pareto": {
            "points": pareto_points,
            "random_points": random_points,
            "n_front": len(front),
            "hypervolume_normalized": summary.get("pareto", {}).get(
                "hypervolume_normalized"
            ),
        },
        "convergence": summary.get("convergence"),
        "surrogate": {
            "points": gp_points,
            "reliability": summary.get("surrogate_reliability"),
            "loo": summary.get("surrogate_loo"),
            "gp": summary.get("gp"),
        },
        "indicators": indicators,
        "random_search_indicators": summary.get("random_search_indicators"),
        "descriptives": summary.get("descriptives"),
        "statistical_tests": summary.get("statistical_tests"),
        "per_variant": summary.get("per_variant"),
        "amortization": summary.get("amortization"),
        "calibration": summary.get("calibration"),
        "trials": summary.get("trials"),
        "thresholds": {
            "eir_improvement": metrics.EIR_IMPROVEMENT_THRESHOLD,
            "eir_substantial": metrics.EIR_SUBSTANTIAL_THRESHOLD,
            "srr": metrics.SRR_THRESHOLD,
            "qrr": metrics.QRR_THRESHOLD,
            "npe": metrics.NPE_RELIABILITY_THRESHOLD,
            "ucr_target": metrics.UCR_TARGET,
        },
        "labels": {
            "conditions": schemas.CONDITION_LABELS,
            "dependent_variables": schemas.DV_LABELS,
        },
        "task_prior": summary.get("task_prior"),
        "scope_warning": summary.get("scope_warning"),
    }
    return payload


def write(result: StudyResult) -> str:
    """Write ``dashboard_data.js`` into the run directory."""
    return result.run_dir.write_dashboard_data(build_payload(result))
