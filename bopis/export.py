"""Chapter 4/5 data export: appendix-shaped CSVs from a finished run.

The run directory already contains one tidy CSV per manuscript table
(``schemas.TABLE_FILES``). Those files are the right shape for software but the
wrong shape for the thesis appendix: the appendix prints wide, human-readable
sheets (one row per prompt, one column per condition, with Mean/SD rows) and a
line-per-iteration calibration log. This module reshapes the tidy artifacts into
that layout and copies the tidy originals alongside them.

Two honesty rules are enforced here:

* **Simulated runs are labelled.** ``EXPORT_MANIFEST.json`` records the backend
  and ``is_measured``; the generated README repeats it. A simulated export must
  never be dropped into Chapter 4 as if it were measured.
* **No fabricated cells.** The per-configuration performance sheet prints
  ``NA`` for per-config CPU/GPU/RAM because the artifacts do not persist those
  per evaluation. Condition-level resource means are available and are printed
  in the condition-level sheet.

Only the standard library is used.
"""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from bopis import artifacts, schemas, stats

#: The manuscript's "Energy > 15%?" flag (Chapter 3, EIR substantial threshold).
ENERGY_FLAG_PERCENT = 15.0

#: Paper-layout file names, keyed by table id.
PAPER_FILES: Dict[str, str] = {
    "B.1": "B1_performance_comparison.csv",
    "B.2": "B2_configuration_evaluation.csv",
    "B.3": "B3_energy_per_prompt.csv",
    "B.4": "B4_speed_per_prompt.csv",
    "B.5": "B5_quality_per_prompt.csv",
    "B.6": "B6_resource_utilization_per_prompt.csv",
    "B.7": "B7_bopis_calibration_log.csv",
    "B.8": "B8_summary_comparison.csv",
    "B.9": "B9_xstar_selection.csv",
    "B.10": "B10_pareto_front.csv",
    "B.11": "B11_surrogate_reliability.csv",
}

#: Tidy copy names under ``tidy/``.
TIDY_FILES: Dict[str, str] = {
    "B.1": "B1_trials.csv",
    "B.2": "B2_config_eval.csv",
    "B.3": "B3_per_prompt_energy.csv",
    "B.4": "B4_per_prompt_speed.csv",
    "B.5": "B5_per_prompt_quality.csv",
    "B.6": "B6_per_prompt_resources.csv",
    "B.7": "B7_bo_log.csv",
    "B.8": "B8_summary.csv",
    "B.9": "B9_selection.csv",
    "B.10": "B10_pareto.csv",
    "B.11": "B11_gp_validation.csv",
}


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #


def _num(value: object, digits: int = 6) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isnan(number) or math.isinf(number):
        return ""
    return f"{number:.{digits}f}"


def _pvalue(value: object) -> str:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return ""
    if math.isnan(number) or math.isinf(number):
        return ""
    if number == 0.0:
        return "< 1e-300"
    if number < 1e-4:
        return f"{number:.6e}"
    return f"{number:.4f}"


def _flag(condition: object) -> str:
    text = str(condition).strip().lower()
    return "Yes" if text in {"true", "1", "yes"} else "No"


def _write_csv(path: str, header: Sequence[str], rows: Iterable[Mapping[str, object]]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(header), restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return path


def _mean_sd(rows: Sequence[Mapping[str, str]], column: str) -> Tuple[str, str]:
    values: List[float] = []
    for row in rows:
        text = str(row.get(column, "")).strip()
        if text == "":
            continue
        try:
            values.append(float(text))
        except ValueError:
            continue
    summary = stats.describe(values)
    return _num(summary.mean), _num(summary.sd)


# --------------------------------------------------------------------------- #
# Trial slicing
# --------------------------------------------------------------------------- #


def _eval_size(rows: Sequence[Mapping[str, str]]) -> int:
    size = 0
    for row in rows:
        try:
            size = max(size, int(row.get("prompt_index", "0")))
        except ValueError:
            continue
    return size or 1


def trial_rows(
    rows: Sequence[Mapping[str, str]], trial: int = 1
) -> List[Mapping[str, str]]:
    """Return one repeat's rows, preserving the run's condition-major order.

    Per-prompt tables are written condition-major, trial-minor: each condition
    has ``eval_size`` rows per trial, so a condition's row ``i`` belongs to
    trial ``i // eval_size + 1``. This reconstructs the trial without a stored
    trial column (amendment planned for the next run).
    """
    if not rows:
        return []
    size = _eval_size(rows)
    seen: Dict[str, int] = {}
    selected: List[Mapping[str, str]] = []
    for row in rows:
        condition = str(row.get("condition", ""))
        index = seen.get(condition, 0)
        seen[condition] = index + 1
        if index // size + 1 == trial:
            selected.append(row)
    return selected


def _pooled(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, str]]:
    """Mean per prompt across every repeat, in the run's original order.

    Trial 1 can pick the same configuration for both methods, which makes the
    single-trial per-prompt comparison degenerate. Pooling per prompt keeps the
    appendix sheet aligned with the pooled descriptive statistics the run
    reports, while still showing 500 prompt rows.
    """
    buckets: Dict[Tuple[str, str], List[Mapping[str, str]]] = {}
    order: List[Tuple[str, str]] = []
    for row in rows:
        key = (str(row.get("condition", "")), str(row.get("prompt_index", "")))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(row)
    pooled: List[Dict[str, str]] = []
    for key in order:
        subset = buckets[key]
        base = dict(subset[0])
        columns = set().union(*(row.keys() for row in subset))
        for column in columns:
            if column == "prompt_index":
                continue
            values: List[float] = []
            complete = True
            for row in subset:
                text = str(row.get(column, "")).strip()
                if text == "":
                    complete = False
                    break
                try:
                    values.append(float(text))
                except ValueError:
                    complete = False
                    break
            if complete and values:
                base[column] = str(sum(values) / len(values))
        try:
            if "tokens_per_s" in base and "s_min" in base:
                base["meets_s_min"] = str(
                    float(base["tokens_per_s"]) >= float(base["s_min"])
                )
            if "quality_f1" in base and "q_min_task" in base:
                base["meets_q_min"] = str(
                    float(base["quality_f1"]) >= float(base["q_min_task"])
                )
        except ValueError:
            pass
        pooled.append(base)
    return pooled


def _select_rows(
    rows: Sequence[Mapping[str, str]], trial: int
) -> List[Mapping[str, str]]:
    """``trial == 0`` pools across repeats; otherwise select that repeat."""
    return _pooled(rows) if trial == 0 else trial_rows(rows, trial)


def _pivot(
    rows: Sequence[Mapping[str, str]], value_column: str
) -> Dict[str, Dict[str, str]]:
    """Map ``prompt_index -> condition -> value`` for the selected rows."""
    pivoted: Dict[str, Dict[str, str]] = {}
    for row in rows:
        prompt = str(row.get("prompt_index", ""))
        condition = str(row.get("condition", ""))
        pivoted.setdefault(prompt, {})[condition] = row.get(value_column, "")
    return pivoted


def _meta_by_prompt(
    rows: Sequence[Mapping[str, str]], column: str
) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for row in rows:
        prompt = str(row.get("prompt_index", ""))
        if prompt not in meta:
            meta[prompt] = row.get(column, "")
    return meta


# --------------------------------------------------------------------------- #
# Per-table builders
# --------------------------------------------------------------------------- #


def _b1_condition(run: artifacts.RunDirectory) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for number, row in enumerate(run.read_table("B.1"), start=1):
        rows.append(
            {
                "Trial No.": number,
                "Config Type": row.get("condition_label", row.get("condition", "")),
                "Energy (J)": _num(row.get("energy_j_mean")),
                "J/token": _num(row.get("j_per_token_mean")),
                "Tokens/sec": _num(row.get("tokens_per_s_mean")),
                "BERTScore F1": _num(row.get("quality_f1_mean")),
                "CPU Util. (%)": _num(row.get("cpu_percent_mean")),
                "GPU Util. (%)": _num(row.get("gpu_percent_mean")),
                "Memory (MB)": _num(row.get("memory_mib_mean")),
            }
        )
    return rows


def _b1_per_config(run: artifacts.RunDirectory) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for number, row in enumerate(run.read_table("B.2"), start=1):
        rows.append(
            {
                "Trial No.": number,
                "Config Type": row.get("method", ""),
                "Energy (J)": _num(row.get("energy_j")),
                "J/token": "NA",
                "Tokens/sec": _num(row.get("tokens_per_s")),
                "BERTScore F1": _num(row.get("quality_f1")),
                "CPU Util. (%)": "NA",
                "GPU Util. (%)": "NA",
                "Memory (MB)": "NA",
            }
        )
    return rows


def _b2(run: artifacts.RunDirectory) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for row in run.read_table("B.2"):
        rows.append(
            {
                "Config ID": row.get("config_id", ""),
                "Token Length": row.get("t_max_gen_tokens", ""),
                "Batch Size": row.get("b_batch_size", ""),
                "Precision": row.get("p_precision", ""),
                "GPU Layers Offloaded": row.get("g_gpu_layers", ""),
                "CPU Threads": row.get("c_cpu_threads", ""),
                "Energy (J)": _num(row.get("energy_j")),
                "Tokens/sec": _num(row.get("tokens_per_s")),
                "BERTScore F1": _num(row.get("quality_f1")),
            }
        )
    return rows


def _b3(
    run: artifacts.RunDirectory, trial: int
) -> Tuple[List[Dict[str, object]], int]:
    primary = _select_rows(run.read_table("B.3"), trial)
    energy = _pivot(primary, "energy_j")
    per_token = _pivot(primary, "j_per_token")
    prompts = sorted(energy, key=lambda value: int(value))
    rows: List[Dict[str, object]] = []
    for prompt in prompts:
        rs_energy = energy[prompt].get("random_search", "")
        bp_energy = energy[prompt].get("bopis", "")
        rs_token = per_token[prompt].get("random_search", "")
        bp_token = per_token[prompt].get("bopis", "")
        flag = ""
        try:
            rs_value = float(rs_token)
            bp_value = float(bp_token)
            if rs_value:
                reduction = (rs_value - bp_value) / rs_value * 100.0
                flag = "Yes" if reduction > ENERGY_FLAG_PERCENT else "No"
        except (TypeError, ValueError):
            flag = ""
        rows.append(
            {
                "Prompt #": prompt,
                "Random Search Total J": _num(rs_energy),
                "Random Search J/Token": _num(rs_token),
                "BOPIS Total J": _num(bp_energy),
                "BOPIS J/Token": _num(bp_token),
                "Flag (> 15%)": flag,
            }
        )
    for label in ("Mean", "SD"):
        mean_row: Dict[str, object] = {"Prompt #": label}
        for display, column in (
            ("Random Search Total J", "energy_j"),
            ("Random Search J/Token", "j_per_token"),
            ("BOPIS Total J", "energy_j"),
            ("BOPIS J/Token", "j_per_token"),
        ):
            condition = "bopis" if display.startswith("BOPIS") else "random_search"
            subset = [r for r in primary if r.get("condition") == condition]
            mean, sd = _mean_sd(subset, column)
            mean_row[display] = mean if label == "Mean" else sd
        mean_row["Flag (> 15%)"] = ""
        rows.append(mean_row)
    return rows, len(prompts)


def _b4(
    run: artifacts.RunDirectory, trial: int
) -> Tuple[List[Dict[str, object]], int]:
    primary = _select_rows(run.read_table("B.4"), trial)
    speed = _pivot(primary, "tokens_per_s")
    meets = _pivot(primary, "meets_s_min")
    tasks = _meta_by_prompt(primary, "task_type")
    xstar = _selected_config_key(run)
    prompts = sorted(speed, key=lambda value: int(value))
    rows: List[Dict[str, object]] = []
    for prompt in prompts:
        rows.append(
            {
                "Prompt #": prompt,
                "Task Type": tasks.get(prompt, ""),
                "Random Search (token/s)": _num(
                    speed[prompt].get("random_search", "")
                ),
                "BOPIS (token/s)": _num(speed[prompt].get("bopis", "")),
                "BOPIS Config Selected": xstar,
                "S >= S_min?": _flag(meets[prompt].get("bopis", "")),
            }
        )
    for label in ("Mean", "SD"):
        row: Dict[str, object] = {"Prompt #": label, "Task Type": ""}
        for display, condition in (
            ("Random Search (token/s)", "random_search"),
            ("BOPIS (token/s)", "bopis"),
        ):
            subset = [r for r in primary if r.get("condition") == condition]
            mean, sd = _mean_sd(subset, "tokens_per_s")
            row[display] = mean if label == "Mean" else sd
        row["BOPIS Config Selected"] = ""
        row["S >= S_min?"] = ""
        rows.append(row)
    return rows, len(prompts)


def _b5(
    run: artifacts.RunDirectory, trial: int
) -> Tuple[List[Dict[str, object]], int]:
    primary = _select_rows(run.read_table("B.5"), trial)
    quality = _pivot(primary, "quality_f1")
    floors = _pivot(primary, "q_min_task")
    meets = _pivot(primary, "meets_q_min")
    tasks = _meta_by_prompt(primary, "task_type")
    prompts = sorted(quality, key=lambda value: int(value))
    rows: List[Dict[str, object]] = []
    for prompt in prompts:
        rows.append(
            {
                "Prompt #": prompt,
                "Task Type": tasks.get(prompt, ""),
                "Random F1": _num(quality[prompt].get("random_search", "")),
                "BOPIS F1": _num(quality[prompt].get("bopis", "")),
                "Q_min(task)": _num(floors[prompt].get("bopis", "")),
                "Q >= Q_min?": _flag(meets[prompt].get("bopis", "")),
            }
        )
    for label in ("Mean", "SD"):
        row: Dict[str, object] = {"Prompt #": label, "Task Type": ""}
        for display, condition in (
            ("Random F1", "random_search"),
            ("BOPIS F1", "bopis"),
        ):
            subset = [r for r in primary if r.get("condition") == condition]
            mean, sd = _mean_sd(subset, "quality_f1")
            row[display] = mean if label == "Mean" else sd
        row["Q_min(task)"] = ""
        row["Q >= Q_min?"] = ""
        rows.append(row)
    return rows, len(prompts)


def _b6(
    run: artifacts.RunDirectory, trial: int
) -> Tuple[List[Dict[str, object]], int]:
    primary = _select_rows(run.read_table("B.6"), trial)
    cpu = _pivot(primary, "cpu_percent")
    gpu = _pivot(primary, "gpu_percent")
    memory = _pivot(primary, "memory_mib")
    prompts = sorted(cpu, key=lambda value: int(value))
    rows: List[Dict[str, object]] = []
    for prompt in prompts:
        rows.append(
            {
                "Prompt #": prompt,
                "RS CPU%": _num(cpu[prompt].get("random_search", "")),
                "RS GPU%": _num(gpu[prompt].get("random_search", "")),
                "RS RAM MB": _num(memory[prompt].get("random_search", "")),
                "BOPIS CPU%": _num(cpu[prompt].get("bopis", "")),
                "BOPIS GPU%": _num(gpu[prompt].get("bopis", "")),
                "BOPIS RAM MB": _num(memory[prompt].get("bopis", "")),
            }
        )
    for label in ("Mean", "SD"):
        row: Dict[str, object] = {"Prompt #": label}
        for display, condition, column in (
            ("RS CPU%", "random_search", "cpu_percent"),
            ("RS GPU%", "random_search", "gpu_percent"),
            ("RS RAM MB", "random_search", "memory_mib"),
            ("BOPIS CPU%", "bopis", "cpu_percent"),
            ("BOPIS GPU%", "bopis", "gpu_percent"),
            ("BOPIS RAM MB", "bopis", "memory_mib"),
        ):
            subset = [r for r in primary if r.get("condition") == condition]
            mean, sd = _mean_sd(subset, column)
            row[display] = mean if label == "Mean" else sd
        rows.append(row)
    return rows, len(prompts)


def _b7(run: artifacts.RunDirectory) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    total_energy = 0.0
    front_size = 0
    for row in run.read_table("B.7"):
        source = row.get("source", "")
        label = "random" if source == "seed" else "BO"
        try:
            total_energy += float(row.get("energy_j") or 0.0)
        except ValueError:
            pass
        if str(row.get("on_pareto_front", "")).strip().lower() in {"true", "1", "yes"}:
            front_size += 1
        config = "t={t},b={b},p={p},g={g},c={c}".format(
            t=row.get("t_max_gen_tokens", ""),
            b=row.get("b_batch_size", ""),
            p=row.get("p_precision", ""),
            g=row.get("g_gpu_layers", ""),
            c=row.get("c_cpu_threads", ""),
        )
        rows.append(
            {
                "Iter.": f"{row.get('iteration', '')} ({label})",
                "Config x (t,b,p,g,c)": config,
                "E measured (J)": _num(row.get("energy_j")),
                "S (tok/s)": _num(row.get("tokens_per_s")),
                "Q (F1)": _num(row.get("quality_f1")),
                "GP mu(x)": _num(row.get("gp_mu")),
                "GP sigma(x)": _num(row.get("gp_sigma")),
                "EI(x)": _num(row.get("expected_improvement")),
                "Pareto Front?": _flag(row.get("on_pareto_front")),
            }
        )
    rows.append(
        {
            "Iter.": "C_boTotal",
            "Config x (t,b,p,g,c)": "",
            "E measured (J)": _num(total_energy),
            "S (tok/s)": "",
            "Q (F1)": "",
            "GP mu(x)": "",
            "GP sigma(x)": "",
            "EI(x)": "",
            "Pareto Front?": str(front_size),
        }
    )
    return rows


def _b8(run: artifacts.RunDirectory, metrics: Mapping[str, object]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    wanted = {"mean": "Mean", "sd": "SD", "min": "Min", "max": "Max"}
    for row in run.read_table("B.8"):
        statistic = str(row.get("statistic", "")).strip().lower()
        if statistic not in wanted:
            continue
        rows.append(
            {
                "DependentVariable": row.get("dv_label", row.get("dependent_variable", "")),
                "Statistic": wanted[statistic],
                "Random Search": _num(row.get("random_search")),
                "BOPIS": _num(row.get("bopis")),
            }
        )
    tests = metrics.get("statistical_tests") or {}
    if isinstance(tests, Mapping):
        for dv, _label, _higher in schemas.DEPENDENT_VARIABLES:
            entry = tests.get(dv) or {}
            wilcoxon = entry.get("wilcoxon") or {} if isinstance(entry, Mapping) else {}
            rows.append(
                {
                    "DependentVariable": schemas.DV_LABELS.get(dv, dv),
                    "Statistic": "Wilcoxon p (RS vs BOPIS)",
                    "Random Search": _pvalue(wilcoxon.get("p_value")),
                    "BOPIS": _pvalue(wilcoxon.get("p_value")),
                }
            )
    return rows


def _selected_config_key(run: artifacts.RunDirectory) -> str:
    for row in run.read_table("B.9"):
        if str(row.get("selected", "")).strip().lower() in {"true", "1", "yes"}:
            return row.get("config", "")
    return ""


# --------------------------------------------------------------------------- #
# Supplementary tables
# --------------------------------------------------------------------------- #


def _merged_prompt_rows(
    run: artifacts.RunDirectory, trial: Optional[int]
) -> List[Dict[str, str]]:
    energy = run.read_table("B.3")
    speed = run.read_table("B.4")
    quality = run.read_table("B.5")
    if trial is not None:
        energy = _select_rows(energy, trial)
        speed = _select_rows(speed, trial)
        quality = _select_rows(quality, trial)

    def index(rows: Sequence[Mapping[str, str]]) -> Dict[Tuple[str, str, int], Mapping[str, str]]:
        seen: Dict[Tuple[str, str], int] = {}
        out: Dict[Tuple[str, str, int], Mapping[str, str]] = {}
        for row in rows:
            key = (str(row.get("condition", "")), str(row.get("prompt_index", "")))
            occurrence = seen.get(key, 0)
            seen[key] = occurrence + 1
            out[(key[0], key[1], occurrence)] = row
        return out

    energy_index = index(energy)
    speed_index = index(speed)
    quality_index = index(quality)
    merged: List[Dict[str, str]] = []
    for key, energy_row in energy_index.items():
        speed_row = speed_index.get(key, {})
        quality_row = quality_index.get(key, {})
        merged.append(
            {
                "condition": key[0],
                "prompt_index": key[1],
                "task_type": energy_row.get("task_type", ""),
                "energy_j": energy_row.get("energy_j", ""),
                "j_per_token": energy_row.get("j_per_token", ""),
                "tokens_per_s": speed_row.get("tokens_per_s", ""),
                "quality_f1": quality_row.get("quality_f1", ""),
            }
        )
    return merged


def _per_task(rows: Sequence[Mapping[str, str]]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str], List[Mapping[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["task_type"], row["condition"]), []).append(row)
    out: List[Dict[str, object]] = []
    for (task, condition), subset in sorted(groups.items()):
        out.append(
            {
                "task_type": task,
                "condition": condition,
                "n": len(subset),
                "energy_j_mean": _num(sum(float(r["energy_j"]) for r in subset) / len(subset)),
                "tokens_per_s_mean": _num(
                    sum(float(r["tokens_per_s"]) for r in subset) / len(subset)
                ),
                "quality_f1_mean": _num(
                    sum(float(r["quality_f1"]) for r in subset) / len(subset)
                ),
            }
        )
    return out


def _per_trial(metrics: Mapping[str, object]) -> List[Dict[str, object]]:
    repeats = metrics.get("repeats") or {}
    trials = repeats.get("per_trial") if isinstance(repeats, Mapping) else None
    if not trials:
        return []
    rows: List[Dict[str, object]] = []
    for entry in trials:
        bopis = entry.get("bopis") or {}
        random_search = entry.get("random_search") or {}
        row: Dict[str, object] = {
            "trial": entry.get("trial", ""),
            "seed": entry.get("seed", ""),
            "ser": _num(entry.get("ser")),
        }
        for prefix, pick in (("bopis", bopis), ("random_search", random_search)):
            row[f"{prefix}_k_star"] = pick.get("k_star", "")
            row[f"{prefix}_config"] = pick.get("config", "")
            row[f"{prefix}_status"] = pick.get("status", "")
            row[f"{prefix}_energy_j"] = _num(pick.get("energy_j"))
            row[f"{prefix}_tokens_per_s"] = _num(pick.get("tokens_per_s"))
            row[f"{prefix}_quality_f1"] = _num(pick.get("quality_f1"))
            row[f"{prefix}_eir_percent"] = _num(pick.get("eir_percent"), 4)
            row[f"{prefix}_srr_percent"] = _num(pick.get("srr_percent"), 4)
            row[f"{prefix}_qrr_percent"] = _num(pick.get("qrr_percent"), 4)
        rows.append(row)
    return rows


def _per_variant(metrics: Mapping[str, object]) -> List[Dict[str, object]]:
    variants = metrics.get("per_variant") or {}
    rows: List[Dict[str, object]] = []
    if not isinstance(variants, Mapping):
        return rows
    for variant, payload in sorted(variants.items()):
        if not isinstance(payload, Mapping):
            continue
        for dv, _label, _higher in schemas.DEPENDENT_VARIABLES:
            description = payload.get(dv)
            if not isinstance(description, Mapping):
                continue
            rows.append(
                {
                    "variant": variant,
                    "dependent_variable": dv,
                    "n": description.get("n", ""),
                    "mean": _num(description.get("mean")),
                    "sd": _num(description.get("sd")),
                    "min": _num(description.get("min")),
                    "max": _num(description.get("max")),
                }
            )
    return rows


def _indicators(metrics: Mapping[str, object]) -> List[Dict[str, object]]:
    bopis = metrics.get("success_indicators") or {}
    random_search = metrics.get("random_search_indicators") or {}
    rows: List[Dict[str, object]] = []
    if not isinstance(bopis, Mapping):
        return rows
    for key in bopis:
        value = bopis[key]
        if isinstance(random_search, Mapping) and key in random_search:
            comparison: object = random_search[key]
        else:
            comparison = ""
        if isinstance(value, bool):
            value = "true" if value else "false"
        if isinstance(comparison, bool):
            comparison = "true" if comparison else "false"
        rows.append(
            {
                "indicator": key,
                "bopis": value,
                "random_search": comparison,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Documentation
# --------------------------------------------------------------------------- #


def _readme(run: artifacts.RunDirectory, manifest: Mapping[str, object], counts: Mapping[str, int]) -> str:
    backend = str((manifest.get("backend") or {}).get("backend", "unknown"))
    measured = backend in {"llama-server"}
    lines = [
        "# Chapter 4/5 data export",
        "",
        f"Run: `{os.path.basename(run.root)}`  ",
        f"Backend: `{backend}`  ",
        f"Measured (not simulated): **{'YES' if measured else 'NO'}**",
        "",
    ]
    if not measured:
        lines += [
            "> WARNING: this run used the simulator. Every energy, speed, and",
            "> resource value below is computed, not measured. Do not report these",
            "> figures in Chapter 4 as empirical results.",
            "",
        ]
    lines += [
        "## Files",
        "",
        "The paper-layout CSVs mirror Appendix B of the manuscript; `tidy/` holds",
        "the run's original long-format tables; `supplementary/` holds derived",
        "tables used by Chapter 4/5 text but not printed as appendix instruments.",
        "",
        "| File | Rows |",
        "| --- | ---: |",
    ]
    for name, count in counts.items():
        lines.append(f"| `{name}` | {count} |")
    lines += [
        "",
        "## Regenerate",
        "",
        "```",
        f"python -m bopis export --run {run.root} --dest docs/chapter4_data",
        "```",
        "",
        "Re-run the same command after the measured run; only the source run changes.",
        "",
    ]
    return "\n".join(lines)


DATA_DICTIONARY = """# Chapter 4/5 data dictionary

All files are UTF-8 CSV unless noted. `NA` means the artifact does not persist
that field for the row; it is never a zero or a guess. Paper-layout tables use
the exact columns printed in Appendix B. `Mean` and `SD` rows are appended to
the per-prompt sheets (sample SD, n-1).

## Paper-layout tables

| File | Appendix table | Source artifact | Notes |
| --- | --- | --- | --- |
| `B1_performance_comparison.csv` | B.1 | `trials.csv` | One row per condition (amendment A-22 reduces the 30-row template to the three conditions actually observed). |
| `B1_performance_per_config.csv` | B.1 | `config_eval.csv` | One row per evaluated configuration; per-config CPU/GPU/RAM are `NA` because only condition-level resources are persisted. |
| `B2_configuration_evaluation.csv` | B.2 | `config_eval.csv` | Search configuration vector with energy, throughput, and BERTScore F1. |
| `B3_energy_per_prompt.csv` | B.3 | `validation/per_prompt_energy.csv` | Primary trial only. Flag `Yes` when BOPIS J/token is more than 15% below random search. |
| `B4_speed_per_prompt.csv` | B.4 | `validation/per_prompt_speed.csv` | Primary trial. `S >= S_min?` is the BOPIS row's own gate result. |
| `B5_quality_per_prompt.csv` | B.5 | `validation/per_prompt_quality.csv` | Primary trial. `Q_min(task)` is the per-task floor used for QRR. |
| `B6_resource_utilization_per_prompt.csv` | B.6 | `validation/per_prompt_resources.csv` | Primary trial. |
| `B7_bopis_calibration_log.csv` | B.7 | `calibration/bo_log.csv` | One row per search iteration; `C_boTotal` row sums search energy and reports front size. |
| `B8_summary_comparison.csv` | B.8 | `summary.csv` + `metrics.json` | Mean/SD/Min/Max per dependent variable plus the paired Wilcoxon p-value (RS vs BOPIS). |
| `B9_xstar_selection.csv` | B.9 | `selection.csv` | Pareto-front ranking and the selected x*. |
| `B10_pareto_front.csv` | B.10 | `pareto.csv` | Evaluated configurations with Pareto membership. |
| `B11_surrogate_reliability.csv` | B.11 | `gp_validation.csv` | GP prediction vs measured energy per validation point. |

## Tidy copies (`tidy/`)

Verbatim copies of the run's B.1-B.11 files, long format, full precision. Use
these for statistics and plots; use the paper-layout files for the appendix.

## Supplementary (`supplementary/`)

| File | Contents |
| --- | --- |
| `per_trial.csv` | One row per repeated seeded trial: seed, k*, selected config, EIR/SRR/QRR, and SER. Empty when `--repeats 1`. |
| `per_task.csv` | Task-type x condition means over the selected trial(s). |
| `per_variant.csv` | Descriptive statistics per precision variant and dependent variable. |
| `indicators.csv` | Success indicators (EIR/SRR/QRR, gates, verdict) for BOPIS and random search. |

## Reading caveats

1. **Per-prompt tables.** By default the 500 rows are the mean per prompt across
   every repeat, which matches the pooled descriptive statistics the run
   reports. Use `--trial N` to export one repeat instead. Repeat 1 can select
   the same configuration for both methods, so a single-trial sheet may show no
   per-prompt difference even when the pooled means differ.
2. **Simulated runs.** Check `EXPORT_MANIFEST.json` (`is_measured`). A simulator
   run is a pipeline check, not an empirical result.
3. **SER < 1** means BOPIS reached its final pick later than random search in
   that trial; do not claim BOPIS is more sample-efficient without checking
   `per_trial.csv`.
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def export_run(
    run_dir: str, out_dir: str, trial: int = 0
) -> Dict[str, object]:
    """Export *run_dir* into *out_dir*; return a summary of what was written."""
    run = artifacts.RunDirectory(run_dir)
    manifest = run.read_json(artifacts.MANIFEST_NAME) or {}
    metrics = run.read_json(artifacts.METRICS_NAME) or {}
    os.makedirs(out_dir, exist_ok=True)
    tidy_dir = os.path.join(out_dir, "tidy")
    supplementary_dir = os.path.join(out_dir, "supplementary")
    os.makedirs(tidy_dir, exist_ok=True)
    os.makedirs(supplementary_dir, exist_ok=True)

    counts: Dict[str, int] = {}

    def emit(table_id: str, header: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
        name = PAPER_FILES[table_id]
        _write_csv(os.path.join(out_dir, name), header, rows)
        counts[name] = len(rows)

    b1 = _b1_condition(run)
    emit("B.1", list(b1[0].keys()) if b1 else [], b1)
    b1_config = _b1_per_config(run)
    _write_csv(
        os.path.join(out_dir, "B1_performance_per_config.csv"),
        list(b1_config[0].keys()) if b1_config else [],
        b1_config,
    )
    counts["B1_performance_per_config.csv"] = len(b1_config)

    b2 = _b2(run)
    emit("B.2", list(b2[0].keys()) if b2 else [], b2)

    b3, prompts = _b3(run, trial)
    emit(
        "B.3",
        [
            "Prompt #",
            "Random Search Total J",
            "Random Search J/Token",
            "BOPIS Total J",
            "BOPIS J/Token",
            "Flag (> 15%)",
        ],
        b3,
    )

    b4, _ = _b4(run, trial)
    emit(
        "B.4",
        [
            "Prompt #",
            "Task Type",
            "Random Search (token/s)",
            "BOPIS (token/s)",
            "BOPIS Config Selected",
            "S >= S_min?",
        ],
        b4,
    )

    b5, _ = _b5(run, trial)
    emit(
        "B.5",
        ["Prompt #", "Task Type", "Random F1", "BOPIS F1", "Q_min(task)", "Q >= Q_min?"],
        b5,
    )

    b6, _ = _b6(run, trial)
    emit(
        "B.6",
        ["Prompt #", "RS CPU%", "RS GPU%", "RS RAM MB", "BOPIS CPU%", "BOPIS GPU%", "BOPIS RAM MB"],
        b6,
    )

    b7 = _b7(run)
    emit(
        "B.7",
        [
            "Iter.",
            "Config x (t,b,p,g,c)",
            "E measured (J)",
            "S (tok/s)",
            "Q (F1)",
            "GP mu(x)",
            "GP sigma(x)",
            "EI(x)",
            "Pareto Front?",
        ],
        b7,
    )

    b8 = _b8(run, metrics)
    emit("B.8", ["DependentVariable", "Statistic", "Random Search", "BOPIS"], b8)

    for table_id in ("B.9", "B.10", "B.11"):
        source = run.table_path(table_id)
        if os.path.exists(source):
            shutil.copyfile(source, os.path.join(out_dir, PAPER_FILES[table_id]))
            counts[PAPER_FILES[table_id]] = len(run.read_table(table_id))
        else:
            _write_csv(os.path.join(out_dir, PAPER_FILES[table_id]), [], [])
            counts[PAPER_FILES[table_id]] = 0

    for table_id, filename in TIDY_FILES.items():
        source = run.table_path(table_id)
        destination = os.path.join(tidy_dir, filename)
        if os.path.exists(source):
            shutil.copyfile(source, destination)
            counts[f"tidy/{filename}"] = len(run.read_table(table_id))
        else:
            counts[f"tidy/{filename}"] = 0

    merged = _merged_prompt_rows(run, trial)
    per_task = _per_task(merged)
    _write_csv(
        os.path.join(supplementary_dir, "per_task.csv"),
        ["task_type", "condition", "n", "energy_j_mean", "tokens_per_s_mean", "quality_f1_mean"],
        per_task,
    )
    counts["supplementary/per_task.csv"] = len(per_task)

    per_trial = _per_trial(metrics)
    if per_trial:
        _write_csv(
            os.path.join(supplementary_dir, "per_trial.csv"),
            list(per_trial[0].keys()),
            per_trial,
        )
    else:
        _write_csv(
            os.path.join(supplementary_dir, "per_trial.csv"),
            ["trial", "seed", "ser"],
            [],
        )
    counts["supplementary/per_trial.csv"] = len(per_trial)

    per_variant = _per_variant(metrics)
    _write_csv(
        os.path.join(supplementary_dir, "per_variant.csv"),
        ["variant", "dependent_variable", "n", "mean", "sd", "min", "max"],
        per_variant,
    )
    counts["supplementary/per_variant.csv"] = len(per_variant)

    indicators = _indicators(metrics)
    _write_csv(
        os.path.join(supplementary_dir, "indicators.csv"),
        ["indicator", "bopis", "random_search"],
        indicators,
    )
    counts["supplementary/indicators.csv"] = len(indicators)

    backend = (manifest.get("backend") or {}).get("backend", "unknown")
    export_manifest = {
        "source_run": os.path.abspath(run.root),
        "source_run_name": os.path.basename(run.root),
        "generated_utc": artifacts.utc_stamp(),
        "bopis_version": manifest.get("bopis_version"),
        "backend": backend,
        "is_measured": backend in {"llama-server"},
        "trial_used": (
            "pooled (per-prompt mean across repeats)" if trial == 0 else trial
        ),
        "per_prompt_rows": prompts,
        "row_counts": counts,
        "git_commit": _git_commit(),
        "note": (
            "Paper-layout tables follow Appendix B; tidy/ holds the run's "
            "original B.1-B.11 files. If is_measured is false, do not report "
            "these values as empirical results."
        ),
    }
    with open(os.path.join(out_dir, "EXPORT_MANIFEST.json"), "w", encoding="utf-8") as handle:
        json.dump(export_manifest, handle, indent=2)
        handle.write("\n")

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as handle:
        handle.write(_readme(run, manifest, counts))
    with open(os.path.join(out_dir, "data_dictionary.md"), "w", encoding="utf-8") as handle:
        handle.write(DATA_DICTIONARY)

    return export_manifest


def _git_commit() -> str:
    """Best-effort HEAD SHA without importing subprocess-heavy helpers."""
    import subprocess

    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, ValueError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""
