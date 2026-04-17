#!/usr/bin/env python3
import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.script_utils import (
    _build_model_and_loss,
    _build_optimizer,
    _parse_csv,
    _parse_float_csv,
    _parse_int_csv,
)
from src.training import train
from src.utils import data_load_and_prep, set_global_seed


def _default_quality_direction(dataset_name: str, quality_metric: str) -> str:
    normalized_metric = quality_metric.lower()
    if "loss" in normalized_metric:
        return "min"
    if dataset_name.lower() == "diabetes" and normalized_metric == "final_test_metric":
        return "min"
    return "max"


def _as_float(value: Any) -> float:
    return float(value)


def _quantiles(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "q10": float(np.quantile(arr, 0.10)),
        "q25": float(np.quantile(arr, 0.25)),
        "q50": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
        "q90": float(np.quantile(arr, 0.90)),
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return

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


def _plot_quality_boxplot_by_sampler(
    rows: list[dict[str, Any]],
    quality_metric: str,
    output_path: Path,
) -> None:
    by_sampler: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_sampler[str(row["sampler_mode"])].append(_as_float(row[quality_metric]))

    samplers = sorted(by_sampler.keys())
    data = [by_sampler[sampler] for sampler in samplers]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(data, labels=samplers, showfliers=True)
    ax.set_title(f"Quality distribution by sampler ({quality_metric})")
    ax.set_xlabel("Sampler")
    ax.set_ylabel(quality_metric)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_bad_run_rate(
    bad_rate_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in bad_rate_rows:
        grouped[str(row["sampler_mode"])].append(float(row["bad_run_rate"]))

    samplers = sorted(grouped.keys())
    means = [float(mean(grouped[sampler])) if grouped[sampler] else 0.0 for sampler in samplers]
    stds = [float(stdev(grouped[sampler])) if len(grouped[sampler]) > 1 else 0.0 for sampler in samplers]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(samplers, means, yerr=stds, capsize=5)
    ax.set_title("Bad-run rate by sampler (mean over shared configs)")
    ax.set_xlabel("Sampler")
    ax.set_ylabel("Bad-run rate")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run identical QA hyperparameter grid on simulated/hybrid/dwave samplers and "
            "analyze run-to-run variance and distribution quality."
        )
    )
    parser.add_argument("--model", required=True, choices=["mlp", "logistic", "svm", "ridge"])
    parser.add_argument("--dataset", required=True, choices=["iris", "wine", "breast_cancer", "digits", "diabetes"])
    parser.add_argument("--samplers", type=_parse_csv, default=["simulated", "hybrid", "dwave"])
    parser.add_argument("--subset-sizes", type=_parse_int_csv, default=[12])
    parser.add_argument("--step-sizes", type=_parse_float_csv, default=[0.05])
    parser.add_argument("--num-reads", type=_parse_int_csv, default=[100])
    parser.add_argument("--seeds", type=_parse_int_csv, default=[7, 21, 42])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", default="32")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default="sampler-transition-analysis")
    parser.add_argument("--run-name-prefix", default="sampler-transition")
    parser.add_argument("--hidden-dims", type=_parse_int_csv, default=[32, 16])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--quality-metric", choices=["final_test_metric", "final_test_loss", "final_train_metric", "final_train_loss"], default="final_test_metric")
    parser.add_argument("--quality-direction", choices=["auto", "max", "min"], default="auto")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-prefix", default="sampler_transition")
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

    runs: list[dict[str, Any]] = []

    grid_configs: list[dict[str, float | int]] = []
    for subset_size in args.subset_sizes:
        for step_size in args.step_sizes:
            for num_reads in args.num_reads:
                grid_configs.append(
                    {
                        "subset_size": int(subset_size),
                        "step_size": float(step_size),
                        "num_reads": int(num_reads),
                    }
                )

    run_index = 0
    for cfg_idx, config in enumerate(grid_configs):
        for sampler_mode in args.samplers:
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
                    optimizer_name="qa",
                    model=model,
                    qa_sampler_mode=sampler_mode,
                    subset_size=int(config["subset_size"]),
                    step_size=float(config["step_size"]),
                    num_reads=int(config["num_reads"]),
                    lr=float(args.lr),
                    damping=float(args.damping),
                )

                run_name = (
                    f"{args.run_name_prefix}-{args.model}-{args.dataset}-"
                    f"cfg{cfg_idx:03d}-{sampler_mode}-seed{seed}-{run_index:04d}"
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
                )

                runs.append(
                    {
                        "config_id": cfg_idx,
                        "model": args.model,
                        "dataset": args.dataset,
                        "sampler_mode": sampler_mode,
                        "seed": int(seed),
                        "subset_size": int(config["subset_size"]),
                        "step_size": float(config["step_size"]),
                        "num_reads": int(config["num_reads"]),
                        "quality_metric": args.quality_metric,
                        "quality_direction": quality_direction,
                        **summary,
                    }
                )
                run_index += 1

    by_sampler: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_sampler[str(row["sampler_mode"])].append(row)

    sampler_summary_rows: list[dict[str, Any]] = []
    for sampler_mode, rows in sorted(by_sampler.items(), key=lambda item: item[0]):
        values = [_as_float(row[args.quality_metric]) for row in rows]
        q = _quantiles(values)
        sampler_summary_rows.append(
            {
                "sampler_mode": sampler_mode,
                "num_runs": len(values),
                "quality_mean": float(mean(values)),
                "quality_std": float(stdev(values)) if len(values) > 1 else 0.0,
                "quality_var": float(np.var(np.asarray(values, dtype=float), ddof=1)) if len(values) > 1 else 0.0,
                "quality_min": float(min(values)),
                "quality_max": float(max(values)),
                **q,
            }
        )

    # Bad-run analysis per shared hyperparameter config.
    bad_run_rows: list[dict[str, Any]] = []
    by_config: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in runs:
        by_config[int(row["config_id"])].append(row)

    for config_id, config_rows in sorted(by_config.items(), key=lambda item: item[0]):
        all_values = [_as_float(row[args.quality_metric]) for row in config_rows]
        if quality_direction == "max":
            threshold = float(np.quantile(np.asarray(all_values, dtype=float), 0.25))
        else:
            threshold = float(np.quantile(np.asarray(all_values, dtype=float), 0.75))

        per_sampler_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in config_rows:
            per_sampler_rows[str(row["sampler_mode"])].append(row)

        for sampler_mode, rows in sorted(per_sampler_rows.items(), key=lambda item: item[0]):
            values = [_as_float(row[args.quality_metric]) for row in rows]
            if quality_direction == "max":
                bad_count = sum(1 for value in values if value <= threshold)
            else:
                bad_count = sum(1 for value in values if value >= threshold)

            bad_run_rows.append(
                {
                    "config_id": config_id,
                    "sampler_mode": sampler_mode,
                    "subset_size": int(rows[0]["subset_size"]),
                    "step_size": float(rows[0]["step_size"]),
                    "num_reads": int(rows[0]["num_reads"]),
                    "quality_threshold": threshold,
                    "num_runs": len(values),
                    "bad_runs": bad_count,
                    "bad_run_rate": float(bad_count / len(values)) if values else 0.0,
                }
            )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.output_prefix}_{timestamp}"

    runs_json = output_dir / f"{stem}_runs.json"
    runs_csv = output_dir / f"{stem}_runs.csv"
    summary_json = output_dir / f"{stem}_sampler_summary.json"
    summary_csv = output_dir / f"{stem}_sampler_summary.csv"
    bad_json = output_dir / f"{stem}_bad_run_analysis.json"
    bad_csv = output_dir / f"{stem}_bad_run_analysis.csv"

    _write_json(runs_json, runs)
    _write_csv(runs_csv, runs)
    _write_json(summary_json, sampler_summary_rows)
    _write_csv(summary_csv, sampler_summary_rows)
    _write_json(bad_json, bad_run_rows)
    _write_csv(bad_csv, bad_run_rows)

    boxplot_path = output_dir / f"{stem}_quality_distribution_boxplot.png"
    bad_rate_plot_path = output_dir / f"{stem}_bad_run_rate.png"
    _plot_quality_boxplot_by_sampler(runs, args.quality_metric, boxplot_path)
    _plot_bad_run_rate(bad_run_rows, bad_rate_plot_path)

    print(
        "Saved sampler-transition artifacts:\n"
        f"- Runs: {runs_json} and {runs_csv}\n"
        f"- Sampler summary: {summary_json} and {summary_csv}\n"
        f"- Bad-run analysis: {bad_json} and {bad_csv}\n"
        f"- Plots: {boxplot_path}, {bad_rate_plot_path}"
    )


if __name__ == "__main__":
    main()
