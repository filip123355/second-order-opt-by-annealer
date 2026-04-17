#!/usr/bin/env python3
import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from src.training import train
from src.utils import data_load_and_prep, set_global_seed
from src.script_utils import (
    _parse_csv,
    _parse_int_csv,
    _parse_float_csv,
    _build_model_and_loss,
    _build_optimizer,
)


def _default_quality_direction(dataset_name: str, quality_metric: str) -> str:
    normalized_metric = quality_metric.lower()
    if "loss" in normalized_metric:
        return "min"
    if dataset_name.lower() == "diabetes" and normalized_metric == "test_metric":
        return "min"
    return "max"

def _build_time_budgets(
    max_elapsed: float,
    requested_grid: list[float] | None,
    num_points: int,
) -> list[float]:
    if requested_grid:
        return sorted({float(value) for value in requested_grid if value >= 0.0})

    if max_elapsed <= 0.0:
        return [0.0]

    if num_points <= 1:
        return [0.0, float(max_elapsed)]

    step = max_elapsed / float(num_points - 1)
    return [float(index * step) for index in range(num_points)]


def _best_so_far(values: list[float], direction: str) -> list[float]:
    out: list[float] = []
    current = -math.inf if direction == "max" else math.inf
    for value in values:
        if direction == "max":
            current = max(current, value)
        else:
            current = min(current, value)
        out.append(current)
    return out


def _quality_at_budget(
    elapsed: list[float],
    quality_values: list[float],
    direction: str,
    budget: float,
) -> float | None:
    if not elapsed:
        return None

    running_best = _best_so_far(quality_values, direction)
    last_value: float | None = None
    for t, q in zip(elapsed, running_best):
        if t <= budget:
            last_value = q
        else:
            break

    return last_value


def _aggregate_budget_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        quality = row.get("quality_at_budget")
        if quality is None:
            continue
        key = (str(row["optimizer_variant"]), float(row["time_budget_sec"]))
        grouped.setdefault(key, []).append(float(quality))

    out: list[dict[str, Any]] = []
    for (optimizer_variant, time_budget_sec), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        summary: dict[str, Any] = {
            "optimizer_variant": optimizer_variant,
            "time_budget_sec": time_budget_sec,
            "num_runs": len(values),
            "quality_mean": float(mean(values)),
            "quality_std": float(stdev(values)) if len(values) > 1 else 0.0,
            "quality_best": float(max(values)),
            "quality_worst": float(min(values)),
        }
        out.append(summary)
    return out


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

    # Rows can have heterogeneous keys (e.g., QA-only telemetry fields), so build a
    # stable superset of columns to avoid DictWriter failures.
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run quality-vs-wall-clock benchmark for multiple optimizers and export run, "
            "timeline and budget-aggregated summaries."
        )
    )
    parser.add_argument("--model", required=True, choices=["mlp", "logistic", "svm", "ridge"])
    parser.add_argument("--dataset", required=True, choices=["iris", "wine", "breast_cancer", "digits", "diabetes"])
    parser.add_argument("--optimizers", type=_parse_csv, default=["qa", "adam", "lbfgs", "newton"])
    parser.add_argument("--qa-samplers", type=_parse_csv, default=["simulated"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", default="32")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seeds", type=_parse_int_csv, default=[7, 21, 42])
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default="quality-vs-wallclock")
    parser.add_argument("--run-name-prefix", default="qvw")
    parser.add_argument("--subset-size", type=int, default=12)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--hidden-dims", type=_parse_int_csv, default=[32, 16])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--quality-metric", choices=["test_metric", "test_loss", "train_metric", "train_loss"], default="test_metric")
    parser.add_argument("--quality-direction", choices=["auto", "max", "min"], default="auto")
    parser.add_argument("--time-grid", type=_parse_float_csv, default=None)
    parser.add_argument("--time-grid-points", type=int, default=15)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-prefix", default="quality_vs_wallclock")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if isinstance(args.batch_size, str) and args.batch_size.lower() != "full":
        args.batch_size = int(args.batch_size)

    quality_direction = (
        _default_quality_direction(args.dataset, args.quality_metric)
        if args.quality_direction == "auto"
        else args.quality_direction
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: list[dict[str, Any]] = []
    timeline_rows: list[dict[str, Any]] = []
    budget_rows_raw: list[dict[str, Any]] = []

    run_index = 0
    max_elapsed_across_runs = 0.0

    for optimizer_name in args.optimizers:
        normalized_optimizer = optimizer_name.lower()
        qa_modes = args.qa_samplers if normalized_optimizer == "qa" else ["not_applicable"]

        for qa_mode in qa_modes:
            for seed in args.seeds:
                set_global_seed(int(seed))

                train_loader, test_loader = data_load_and_prep(
                    dataset_name=args.dataset,
                    test_size=args.test_size,
                    random_state=int(seed),
                    batch_size=args.batch_size,
                    shuffle=True,
                )

                model, loss_fn = _build_model_and_loss(
                    model_name=args.model,
                    dataset_name=args.dataset,
                    train_loader=train_loader,
                    hidden_dims=args.hidden_dims,
                )

                optimizer = _build_optimizer(
                    optimizer_name=normalized_optimizer,
                    model=model,
                    qa_sampler_mode=qa_mode,
                    subset_size=int(args.subset_size),
                    step_size=float(args.step_size),
                    num_reads=int(args.num_reads),
                    lr=float(args.lr),
                    damping=float(args.damping),
                )

                optimizer_variant = f"qa:{qa_mode}" if normalized_optimizer == "qa" else normalized_optimizer
                run_name = (
                    f"{args.run_name_prefix}-{args.model}-{args.dataset}-"
                    f"{optimizer_variant}-seed{seed}-{run_index:03d}"
                )

                summary = train(
                    model=model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    loss_fn=loss_fn,
                    c_device=args.device,
                    optimizer=optimizer,
                    epochs=args.epochs,
                    tracking_uri=args.tracking_uri,
                    experiment_name=f"{args.experiment_name}-{args.model}-{args.dataset}",
                    log_batch_metrics=True,
                    run_name=run_name,
                    verbose=not args.quiet,
                    seed=int(seed),
                    collect_epoch_history=True,
                )

                epoch_history_raw = summary.pop("epoch_history", [])
                epoch_history = list(epoch_history_raw) if isinstance(epoch_history_raw, list) else []
                run_key = f"{optimizer_variant}-seed{seed}-{run_index:03d}"

                training_time_sec = float(summary.get("training_time_sec", 0.0))
                max_elapsed_across_runs = max(max_elapsed_across_runs, training_time_sec)

                run_record: dict[str, Any] = {
                    "run_key": run_key,
                    "model": args.model,
                    "dataset": args.dataset,
                    "optimizer": normalized_optimizer,
                    "optimizer_variant": optimizer_variant,
                    "qa_sampler_mode": qa_mode,
                    "seed": int(seed),
                    "quality_metric": args.quality_metric,
                    "quality_direction": quality_direction,
                    **summary,
                }
                run_summaries.append(run_record)

                for epoch_entry in epoch_history:
                    row = {
                        "run_key": run_key,
                        "model": args.model,
                        "dataset": args.dataset,
                        "optimizer": normalized_optimizer,
                        "optimizer_variant": optimizer_variant,
                        "qa_sampler_mode": qa_mode,
                        "seed": int(seed),
                        **epoch_entry,
                    }
                    timeline_rows.append(row)

                run_index += 1

    time_budgets = _build_time_budgets(
        max_elapsed=max_elapsed_across_runs,
        requested_grid=args.time_grid,
        num_points=int(args.time_grid_points),
    )

    # Build per-run quality-at-budget rows.
    by_run: dict[str, list[dict[str, Any]]] = {}
    run_meta: dict[str, dict[str, Any]] = {}
    for row in timeline_rows:
        key = str(row["run_key"])
        by_run.setdefault(key, []).append(row)
        if key not in run_meta:
            run_meta[key] = {
                "optimizer_variant": row["optimizer_variant"],
                "optimizer": row["optimizer"],
                "qa_sampler_mode": row["qa_sampler_mode"],
                "seed": row["seed"],
                "quality_metric": args.quality_metric,
                "quality_direction": quality_direction,
            }

    for run_key, rows in by_run.items():
        ordered = sorted(rows, key=lambda item: float(item["elapsed_time_sec"]))
        elapsed = [float(item["elapsed_time_sec"]) for item in ordered]
        quality_values = [float(item[args.quality_metric]) for item in ordered]

        for budget in time_budgets:
            quality = _quality_at_budget(
                elapsed=elapsed,
                quality_values=quality_values,
                direction=quality_direction,
                budget=float(budget),
            )
            budget_rows_raw.append(
                {
                    "run_key": run_key,
                    "optimizer_variant": run_meta[run_key]["optimizer_variant"],
                    "optimizer": run_meta[run_key]["optimizer"],
                    "qa_sampler_mode": run_meta[run_key]["qa_sampler_mode"],
                    "seed": run_meta[run_key]["seed"],
                    "quality_metric": args.quality_metric,
                    "quality_direction": quality_direction,
                    "time_budget_sec": float(budget),
                    "quality_at_budget": quality,
                }
            )

    budget_aggregated = _aggregate_budget_rows(budget_rows_raw)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.output_prefix}_{timestamp}"

    run_json = output_dir / f"{stem}_runs.json"
    run_csv = output_dir / f"{stem}_runs.csv"
    timeline_json = output_dir / f"{stem}_timeline.json"
    timeline_csv = output_dir / f"{stem}_timeline.csv"
    budget_raw_json = output_dir / f"{stem}_budget_raw.json"
    budget_raw_csv = output_dir / f"{stem}_budget_raw.csv"
    budget_summary_json = output_dir / f"{stem}_budget_summary.json"
    budget_summary_csv = output_dir / f"{stem}_budget_summary.csv"

    _write_json(run_json, run_summaries)
    _write_csv(run_csv, run_summaries)
    _write_json(timeline_json, timeline_rows)
    _write_csv(timeline_csv, timeline_rows)
    _write_json(budget_raw_json, budget_rows_raw)
    _write_csv(budget_raw_csv, budget_rows_raw)
    _write_json(budget_summary_json, budget_aggregated)
    _write_csv(budget_summary_csv, budget_aggregated)

    print(
        "Saved quality-vs-wall-clock artifacts:\n"
        f"- Runs: {run_json} and {run_csv}\n"
        f"- Timeline: {timeline_json} and {timeline_csv}\n"
        f"- Budget raw: {budget_raw_json} and {budget_raw_csv}\n"
        f"- Budget summary: {budget_summary_json} and {budget_summary_csv}"
    )


if __name__ == "__main__":
    main()
