"""Build UREC per-data-collection tables from the authoritative Chapter 4/5 run."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

SCRIPT = Path(__file__).resolve()
ROOT = next(
    parent
    for parent in SCRIPT.parents
    if (parent / "bopis").is_dir() and (parent / "runs").is_dir()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bopis import classify, classify_trained, dataset, metrics as metrics_module, tasks

RUN = ROOT / "runs" / "20261107T125343Z_real-full"
CHAPTER = ROOT / "docs" / "chapter4_data"
TIDY = CHAPTER / "tidy"
DEST = ROOT / "docs" / "UREC_per_data_collection"
DOLLY = ROOT / "data" / "databricks-dolly-15k.jsonl"

CONDITION_ORDER = {"unoptimized": 0, "random_search": 1, "bopis": 2}
CONDITION_LABEL = {
    "unoptimized": "Unoptimized Default",
    "random_search": "Random Search",
    "bopis": "BOPIS",
}
MODEL_LABEL = {
    "rule_baseline": "Rule Baseline",
    "learned_naive_bayes": "Learned Naive Bayes",
}
B10_LABEL_ORDER = (
    "open_qa",
    "closed_qa",
    "summarization",
    "information_extraction",
    "general_qa",
    "classification",
    "brainstorming",
    "creative_writing",
)
BOPIS_METRICS = [
    ("energy_j", "Energy (J)"),
    ("j_per_token", "Energy per Token (J/token)"),
    ("tokens_per_s", "Speed (tokens/s)"),
    ("quality_f1", "Quality (BERTScore F1)"),
    ("cpu_percent", "CPU Utilization (%)"),
    ("gpu_percent", "GPU Utilization (%)"),
    ("memory_mib", "System Memory (MiB)"),
    ("vram_mib", "VRAM (MiB)"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def config_key(
    m_model: str, t: object, b: object, p: object, g: object, c: object
) -> str:
    return f"{m_model}_t{t}_b{b}_{p}_g{g}_c{c}"


def build_bopis_table(
    metrics: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    sources = OrderedDict(
        [
            ("energy", TIDY / "B3_per_prompt_energy.csv"),
            ("speed", TIDY / "B4_per_prompt_speed.csv"),
            ("quality", TIDY / "B5_per_prompt_quality.csv"),
            ("resources", TIDY / "B6_per_prompt_resources.csv"),
        ]
    )
    key_fields = ("prompt_index", "prompt_id", "task_type", "condition")
    merged: OrderedDict[tuple[str, str, str, str], dict[str, str]] = OrderedDict()
    for path in sources.values():
        rows = read_csv(path)
        fields = list(rows[0].keys())
        for row in rows:
            key = tuple(row[field] for field in key_fields)
            target = merged.setdefault(key, {field: row[field] for field in key_fields})
            for field in fields:
                if field not in target:
                    target[field] = row[field]

    configs: dict[str, str] = {}
    for trial in metrics["trials"]:
        condition = str(trial["condition"])
        configs[condition] = config_key(
            str(trial["m_model"]),
            trial["t_max_gen_tokens"],
            trial["b_batch_size"],
            trial["p_precision"],
            trial["g_gpu_layers"],
            trial["c_cpu_threads"],
        )

    rows: list[dict[str, object]] = []
    for target in merged.values():
        condition = target["condition"]
        rows.append(
            {
                "test_item_no": int(target["prompt_index"]),
                "prompt_index": target["prompt_index"],
                "prompt_id": target["prompt_id"],
                "task_type": target["task_type"],
                "condition": condition,
                "condition_label": CONDITION_LABEL.get(condition, condition),
                "configuration": configs.get(condition, ""),
                "energy_j": target.get("energy_j", ""),
                "j_per_token": target.get("j_per_token", ""),
                "n_generated_tokens": target.get("n_generated_tokens", ""),
                "prefill_energy_j": target.get("prefill_energy_j", ""),
                "decode_energy_j": target.get("decode_energy_j", ""),
                "tokens_per_s": target.get("tokens_per_s", ""),
                "decode_tokens_per_s": target.get("decode_tokens_per_s", ""),
                "wall_s": target.get("wall_s", ""),
                "quality_f1": target.get("quality_f1", ""),
                "quality_precision": target.get("quality_precision", ""),
                "quality_recall": target.get("quality_recall", ""),
                "cpu_percent": target.get("cpu_percent", ""),
                "process_cpu_percent": target.get("process_cpu_percent", ""),
                "gpu_percent": target.get("gpu_percent", ""),
                "memory_mib": target.get("memory_mib", ""),
                "vram_mib": target.get("vram_mib", ""),
                "n_threads": target.get("n_threads", ""),
                "meets_s_min": target.get("meets_s_min", ""),
                "meets_q_min": target.get("meets_q_min", ""),
                "s_min": target.get("s_min", ""),
                "q_min_task": target.get("q_min_task", ""),
                "energy_method": target.get("energy_method", ""),
                "energy_scope": target.get("energy_scope", ""),
                "energy_basis": target.get("energy_basis", ""),
                "scorer": target.get("scorer", ""),
            }
        )
    rows.sort(
        key=lambda row: (
            int(row["prompt_index"]),
            CONDITION_ORDER.get(str(row["condition"]), 99),
        )
    )

    summary: list[dict[str, object]] = []
    for condition in sorted(CONDITION_ORDER, key=CONDITION_ORDER.get):
        subset = [row for row in rows if row["condition"] == condition]
        for field, label in BOPIS_METRICS:
            values = [
                float(row[field])
                for row in subset
                if row.get(field) not in ("", None, "NA")
            ]
            if not values:
                continue
            summary.append(
                {
                    "condition": condition,
                    "condition_label": CONDITION_LABEL[condition],
                    "metric": label,
                    "field": field,
                    "n": len(values),
                    "mean": f"{statistics.mean(values):.6f}",
                    "sd": (
                        f"{statistics.stdev(values):.6f}"
                        if len(values) > 1
                        else "NA"
                    ),
                    "min": f"{min(values):.6f}",
                    "max": f"{max(values):.6f}",
                }
            )
    return rows, summary


def build_task_category_table() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, classify_trained.Scores],
    classify_trained.TrainingReport,
]:
    raw = dataset.load_jsonl(str(DOLLY))
    rows: list[dict[str, object]] = []
    for index, row in enumerate(raw, start=1):
        item = dict(row)
        item["dolly_index"] = index
        rows.append(item)

    model, report = classify_trained.train_and_evaluate(rows)
    _, _, test_rows = classify_trained.stratified_split(rows)
    if len(test_rows) != report.n_test:
        raise RuntimeError("split size changed between training and export")

    per_item: list[dict[str, object]] = []
    learned_truths: list[str] = []
    learned_predictions: list[str] = []
    rule_truths: list[str] = []
    rule_predictions: list[str] = []
    for item_no, row in enumerate(test_rows, start=1):
        actual = row["category"]
        instruction = row.get("instruction", "")
        context = row.get("context", "")
        tokens = classify_trained.tokenize(instruction, context)
        learned = classify_trained.predict_prompt(model, instruction, context)
        rule = classify.classify_prompt(instruction, context)
        learned_truths.append(actual)
        learned_predictions.append(learned.task_key)
        rule_truths.append(actual)
        rule_predictions.append(rule.task_key)
        for model_key, prediction in (
            ("rule_baseline", rule),
            ("learned_naive_bayes", learned),
        ):
            per_item.append(
                {
                    "test_item_no": item_no,
                    "dolly_index": row["dolly_index"],
                    "prompt_id": f"dolly-{int(row['dolly_index']):05d}",
                    "instruction": instruction,
                    "context": context,
                    "context_present": "Yes" if context.strip() else "No",
                    "actual_category": actual,
                    "model": model_key,
                    "model_label": MODEL_LABEL[model_key],
                    "predicted_category": prediction.task_key,
                    "confidence": f"{prediction.confidence:.6f}",
                    "exact_correct": "Yes" if prediction.task_key == actual else "No",
                    "qa_merged_correct": "Yes"
                    if (
                        prediction.task_key == actual
                        or {prediction.task_key, actual} == {"open_qa", "general_qa"}
                    )
                    else "No",
                    "sensitivity_tier_correct": "Yes"
                    if (
                        tasks.TASK_BY_KEY[prediction.task_key].sensitivity
                        == tasks.TASK_BY_KEY[actual].sensitivity
                    )
                    else "No",
                }
            )
    learned_scores = classify_trained.score_predictions(
        learned_truths, learned_predictions
    )
    rule_scores = classify_trained.score_predictions(rule_truths, rule_predictions)
    scores = {"rule_baseline": rule_scores, "learned_naive_bayes": learned_scores}

    labels = list(B10_LABEL_ORDER)

    def counts(
        confusion: dict[str, dict[str, int]], label: str
    ) -> tuple[int, int, int]:
        tp = confusion[label].get(label, 0)
        actual_n = sum(confusion[label].values())
        predicted_n = sum(confusion[truth].get(label, 0) for truth in labels)
        return tp, actual_n, predicted_n

    def macro(
        confusion: dict[str, dict[str, int]]
    ) -> tuple[float, float, float]:
        precisions: list[float] = []
        recalls: list[float] = []
        f1s: list[float] = []
        for label in labels:
            tp, actual_n, predicted_n = counts(confusion, label)
            fp = predicted_n - tp
            fn = actual_n - tp
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        return (
            sum(precisions) / len(precisions),
            sum(recalls) / len(recalls),
            sum(f1s) / len(f1s),
        )

    def metrics_row(
        model_key: str, score: classify_trained.Scores
    ) -> dict[str, object]:
        macro_precision, macro_recall, macro_f1 = macro(score.confusion)
        return {
            "model": model_key,
            "model_label": MODEL_LABEL[model_key],
            "n": score.n,
            "exact_correct": score.exact,
            "accuracy": f"{score.exact_accuracy:.6f}",
            "qa_merged_correct": score.qa_merged,
            "qa_merged_accuracy": f"{score.qa_merged_accuracy:.6f}",
            "sensitivity_tier_correct": score.sensitivity,
            "sensitivity_tier_accuracy": f"{score.sensitivity_accuracy:.6f}",
            "macro_precision": f"{macro_precision:.6f}",
            "macro_recall": f"{macro_recall:.6f}",
            "macro_f1": f"{macro_f1:.6f}",
        }

    metrics_rows = [
        metrics_row("rule_baseline", rule_scores),
        metrics_row("learned_naive_bayes", learned_scores),
    ]

    per_class: list[dict[str, object]] = []
    for model_key in ("rule_baseline", "learned_naive_bayes"):
        score = scores[model_key]
        for label in labels:
            tp, actual_n, predicted_n = counts(score.confusion, label)
            fp = predicted_n - tp
            fn = actual_n - tp
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall)
                else 0.0
            )
            per_class.append(
                {
                    "model": model_key,
                    "model_label": MODEL_LABEL[model_key],
                    "category": label,
                    "actual_n": actual_n,
                    "predicted_n": predicted_n,
                    "true_positive": tp,
                    "false_positive": fp,
                    "false_negative": fn,
                    "precision": f"{precision:.6f}",
                    "recall": f"{recall:.6f}",
                    "f1": f"{f1:.6f}",
                }
            )

    confusion_rows: list[dict[str, object]] = []
    for actual in labels:
        row: dict[str, object] = {"actual_category": actual}
        for predicted in labels:
            row[predicted] = learned_scores.confusion[actual].get(predicted, 0)
        row["actual_n"] = learned_scores.per_category[actual][1]
        confusion_rows.append(row)

    return per_item, metrics_rows, per_class, confusion_rows, scores, report


def build_b9_table(metrics: dict[str, object]) -> list[dict[str, object]]:
    reference = metrics["search_reference"]
    s_ref = float(reference["tokens_per_s"])
    q_ref = float(reference["quality_f1"])
    rows = read_csv(TIDY / "B7_bo_log.csv")
    output: list[dict[str, object]] = []
    for row in rows:
        energy = float(row["energy_j"])
        speed = float(row["tokens_per_s"])
        quality = float(row["quality_f1"])
        srr = speed / s_ref * 100.0
        qrr = quality / q_ref * 100.0
        output.append(
            {
                "test_item_no": int(row["iteration"]),
                "iteration": int(row["iteration"]),
                "source": row["source"],
                "configuration": row["config"],
                "m_model": row["m_model"],
                "t_max_gen_tokens": row["t_max_gen_tokens"],
                "b_batch_size": row["b_batch_size"],
                "p_precision": row["p_precision"],
                "g_gpu_layers": row["g_gpu_layers"],
                "c_cpu_threads": row["c_cpu_threads"],
                "energy_j": f"{energy:.6f}",
                "tokens_per_s": f"{speed:.6f}",
                "quality_f1": f"{quality:.6f}",
                "srr_percent": f"{srr:.6f}",
                "qrr_percent": f"{qrr:.6f}",
                "meets_srr_gate": "Yes"
                if srr >= metrics_module.SRR_THRESHOLD
                else "No",
                "meets_qrr_gate": "Yes"
                if qrr >= metrics_module.QRR_THRESHOLD
                else "No",
                "on_pareto_front": row["on_pareto_front"],
                "gp_mu_energy_j": row.get("gp_mu", ""),
                "gp_sigma_energy_j": row.get("gp_sigma", ""),
                "expected_improvement": row.get("expected_improvement", ""),
                "p_feasible": row.get("p_feasible", ""),
                "best_energy_so_far": row.get("best_energy_so_far", ""),
                "delta_energy_j": row.get("delta_energy", ""),
            }
        )
    return output


def protected_b10_matrix() -> dict[str, dict[str, int]]:
    rows = read_csv(CHAPTER / "table_b10_task_category_confusion_matrix.csv")
    matrix: dict[str, dict[str, int]] = {}
    for row in rows:
        actual = row["actual_category"]
        if actual not in tasks.TASK_BY_KEY:
            continue
        matrix[actual] = {
            label: int(row[label])
            for label in tasks.TASK_BY_KEY
            if row.get(label) not in ("", None)
        }
    return matrix


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((RUN / "metrics.json").read_text(encoding="utf-8"))

    bopis_rows, bopis_summary = build_bopis_table(metrics)
    write_csv(
        DEST / "UREC_Table2B_bopis_per_test_input.csv",
        [
            "test_item_no",
            "prompt_index",
            "prompt_id",
            "task_type",
            "condition",
            "condition_label",
            "configuration",
            "energy_j",
            "j_per_token",
            "n_generated_tokens",
            "prefill_energy_j",
            "decode_energy_j",
            "tokens_per_s",
            "decode_tokens_per_s",
            "wall_s",
            "quality_f1",
            "quality_precision",
            "quality_recall",
            "cpu_percent",
            "process_cpu_percent",
            "gpu_percent",
            "memory_mib",
            "vram_mib",
            "n_threads",
            "meets_s_min",
            "meets_q_min",
            "s_min",
            "q_min_task",
            "energy_method",
            "energy_scope",
            "energy_basis",
            "scorer",
        ],
        bopis_rows,
    )
    write_csv(
        DEST / "UREC_Table3D_bopis_condition_summary.csv",
        [
            "condition",
            "condition_label",
            "metric",
            "field",
            "n",
            "mean",
            "sd",
            "min",
            "max",
        ],
        bopis_summary,
    )

    (
        task_rows,
        task_metrics,
        task_per_class,
        task_confusion,
        task_scores,
        training_report,
    ) = build_task_category_table()
    write_csv(
        DEST / "UREC_Table2A_task_category_per_test_input.csv",
        [
            "test_item_no",
            "dolly_index",
            "prompt_id",
            "instruction",
            "context",
            "context_present",
            "actual_category",
            "model",
            "model_label",
            "predicted_category",
            "confidence",
            "exact_correct",
            "qa_merged_correct",
            "sensitivity_tier_correct",
        ],
        task_rows,
    )
    by_item: dict[int, dict[str, dict[str, object]]] = {}
    for row in task_rows:
        by_item.setdefault(int(row["test_item_no"]), {})[str(row["model"])] = row
    b10a_rows: list[dict[str, object]] = []
    for item_no in sorted(by_item):
        learned_row = by_item[item_no]["learned_naive_bayes"]
        rule_row = by_item[item_no]["rule_baseline"]
        learned_prediction = str(learned_row["predicted_category"])
        rule_prediction = str(rule_row["predicted_category"])
        b10a_rows.append(
            {
                "test_item_no": item_no,
                "dolly_index": learned_row["dolly_index"],
                "prompt_id": learned_row["prompt_id"],
                "instruction": learned_row["instruction"],
                "context": learned_row["context"],
                "context_present": learned_row["context_present"],
                "actual_category": learned_row["actual_category"],
                "learned_predicted_category": learned_prediction,
                "learned_confidence": learned_row["confidence"],
                "learned_correct": learned_row["exact_correct"],
                "rule_predicted_category": rule_prediction,
                "rule_confidence": rule_row["confidence"],
                "rule_correct": rule_row["exact_correct"],
                "models_agree": "Yes"
                if learned_prediction == rule_prediction
                else "No",
            }
        )
    write_csv(
        DEST / "UREC_Table2A_B10a_task_category_per_test_input.csv",
        [
            "test_item_no",
            "dolly_index",
            "prompt_id",
            "instruction",
            "context",
            "context_present",
            "actual_category",
            "learned_predicted_category",
            "learned_confidence",
            "learned_correct",
            "rule_predicted_category",
            "rule_confidence",
            "rule_correct",
            "models_agree",
        ],
        b10a_rows,
    )
    write_csv(
        DEST / "UREC_Table3A_task_category_metrics_by_model.csv",
        [
            "model",
            "model_label",
            "n",
            "exact_correct",
            "accuracy",
            "qa_merged_correct",
            "qa_merged_accuracy",
            "sensitivity_tier_correct",
            "sensitivity_tier_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ],
        task_metrics,
    )
    write_csv(
        DEST / "UREC_Table3B_task_category_per_class_metrics.csv",
        [
            "model",
            "model_label",
            "category",
            "actual_n",
            "predicted_n",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
        ],
        task_per_class,
    )
    labels = list(B10_LABEL_ORDER)
    write_csv(
        DEST / "UREC_Table3C_task_category_confusion_matrix_learned.csv",
        ["actual_category", *labels, "actual_n"],
        task_confusion,
    )

    b9_rows = build_b9_table(metrics)
    write_csv(
        DEST / "UREC_Table2C_b9_gate_viability_per_iteration.csv",
        [
            "test_item_no",
            "iteration",
            "source",
            "configuration",
            "m_model",
            "t_max_gen_tokens",
            "b_batch_size",
            "p_precision",
            "g_gpu_layers",
            "c_cpu_threads",
            "energy_j",
            "tokens_per_s",
            "quality_f1",
            "srr_percent",
            "qrr_percent",
            "meets_srr_gate",
            "meets_qrr_gate",
            "on_pareto_front",
            "gp_mu_energy_j",
            "gp_sigma_energy_j",
            "expected_improvement",
            "p_feasible",
            "best_energy_so_far",
            "delta_energy_j",
        ],
        b9_rows,
    )

    learned = task_scores["learned_naive_bayes"]
    rule = task_scores["rule_baseline"]
    protected = protected_b10_matrix()
    mismatches: list[str] = []
    for actual in labels:
        for predicted in labels:
            expected = protected.get(actual, {}).get(predicted, 0)
            observed = learned.confusion[actual].get(predicted, 0)
            if expected != observed:
                mismatches.append(
                    f"{actual}->{predicted}: protected={expected} "
                    f"observed={observed}"
                )

    macro_precision = float(task_metrics[1]["macro_precision"])
    macro_recall = float(task_metrics[1]["macro_recall"])
    macro_f1 = float(task_metrics[1]["macro_f1"])
    b10_rows = read_csv(CHAPTER / "table_b10_task_category_confusion_matrix.csv")
    protected_accuracy = None
    protected_macro_f1 = None
    for row in b10_rows:
        metric_name = row.get("actual_category")
        if metric_name == "overall_exact_accuracy":
            protected_accuracy = float(row["closed_qa"])
        if metric_name == "macro_f1":
            protected_macro_f1 = float(row["closed_qa"])
    if protected_accuracy is not None and abs(
        protected_accuracy - learned.exact_accuracy
    ) > 5e-6:
        mismatches.append(
            f"accuracy: protected={protected_accuracy} "
            f"observed={learned.exact_accuracy}"
        )
    if protected_macro_f1 is not None and abs(protected_macro_f1 - macro_f1) > 5e-6:
        mismatches.append(
            f"macro_f1: protected={protected_macro_f1} observed={macro_f1}"
        )

    lines = [
        "UREC per-data-collection export validation",
        "=" * 48,
        f"source run: {RUN.name}",
        f"dolly rows: {len(dataset.load_jsonl(str(DOLLY)))}",
        f"split: train {training_report.n_train} / "
        f"validation {training_report.n_validation} / "
        f"test {training_report.n_test}",
        "",
        "B10 task-category reproduction",
        f"  learned exact: {learned.exact}/{learned.n} = {learned.exact_accuracy:.6f}",
        f"  protected accuracy: {protected_accuracy}",
        f"  learned macro precision: {macro_precision:.6f}",
        f"  learned macro recall: {macro_recall:.6f}",
        f"  learned macro F1: {macro_f1:.6f}",
        f"  protected macro F1: {protected_macro_f1}",
        f"  rule exact: {rule.exact}/{rule.n} = {rule.exact_accuracy:.6f}",
        f"  confusion mismatches: {len(mismatches)}",
    ]
    lines.extend(f"    {item}" for item in mismatches)
    lines += [
        "",
        "Row counts",
        f"  B.10a wide per-test-input rows: {len(b10a_rows)}",
        f"  task-category long rows: {len(task_rows)} "
        f"(2 models x {learned.n} test items)",
        f"  bopis per-prompt rows: {len(bopis_rows)}",
        f"  b9 iteration rows: {len(b9_rows)}",
        "",
        "Source hashes (sha256)",
    ]
    for path in [
        TIDY / "B3_per_prompt_energy.csv",
        TIDY / "B4_per_prompt_speed.csv",
        TIDY / "B5_per_prompt_quality.csv",
        TIDY / "B6_per_prompt_resources.csv",
        TIDY / "B7_bo_log.csv",
        CHAPTER / "table_b10_task_category_confusion_matrix.csv",
        CHAPTER / "table_b9_gate_viability_confusion_matrix.csv",
        DOLLY,
    ]:
        lines.append(f"  {path.relative_to(ROOT)}  {sha256(path)}")
    lines += ["", "Output hashes (sha256)"]
    for path in sorted(DEST.glob("UREC_*.csv")):
        lines.append(f"  {path.name}  {sha256(path)}")

    report_text = "\n".join(lines) + "\n"
    (DEST / "validation_report.txt").write_text(
        report_text, encoding="utf-8", newline="\n"
    )
    print(report_text)
    return 0 if not mismatches else 1


if __name__ == "__main__":
    raise SystemExit(main())
