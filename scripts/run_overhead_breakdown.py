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

from src.script_utils import (
    _build_model_and_loss,
    _build_optimizer,
    _parse_csv,
    _parse_int_csv,
)
from src.training import train
from src.utils import data_load_and_prep, set_global_seed


_COMPONENTS = [
    "build_bqm_time_sec",
    "transfer_time_sec",
    "sampling_time_sec",
    "update_time_sec",
]


def _mean_epoch_metric(epoch_history: list[dict[str, Any]], key: str) -> float:
    values = [float(epoch[key]) for epoch in epoch_history if key in epoch]
    if not values:
        return 0.0
    return float(mean(values))


def _to_percent(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return 100.0 * numerator / denominator


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


def _plot_sampler_breakdown(
    sampler_mode: str,
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    ordered = sorted(rows, key=lambda row: int(row["subset_size"]))
    x_values = [int(row["subset_size"]) for row in ordered]

    build_share = [float(row["build_bqm_share_pct_mean"]) for row in ordered]
    transfer_share = [float(row["transfer_share_pct_mean"]) for row in ordered]
    sampling_share = [float(row["sampling_share_pct_mean"]) for row in ordered]
    update_share = [float(row["update_share_pct_mean"]) for row in ordered]

    fig, ax = plt.subplots(figsize=(10, 6))

    width = 0.65
    ax.bar(x_values, build_share, width=width, label="build_bqm")
    ax.bar(x_values, transfer_share, width=width, bottom=build_share, label="transfer")

    build_plus_transfer = [b + t for b, t in zip(build_share, transfer_share)]
    ax.bar(x_values, sampling_share, width=width, bottom=build_plus_transfer, label="sampling")

    build_transfer_sampling = [b + t + s for b, t, s in zip(build_share, transfer_share, sampling_share)]
    ax.bar(x_values, update_share, width=width, bottom=build_transfer_sampling, label="update")

    ax.set_title(f"Overhead share vs problem size (sampler={sampler_mode})")
    ax.set_xlabel("Problem size (subset_size)")
    ax.set_ylabel("Time share [%]")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run communication-overhead experiment for QA optimizer: split per-step time into "
            "build BQM, transfer, sampling, and update; then plot percentage share vs subset_size."
        )
    )
    parser.add_argument("--model", required=True, choices=["mlp", "logistic", "svm", "ridge"])
    parser.add_argument("--dataset", required=True, choices=["iris", "wine", "breast_cancer", "digits", "diabetes"])
    parser.add_argument("--qa-samplers", type=_parse_csv, default=["simulated"])
    parser.add_argument("--subset-sizes", type=_parse_int_csv, default=[6, 12, 24, 36])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", default="32")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seeds", type=_parse_int_csv, default=[7, 21, 42])
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default="qa-overhead-breakdown")
    parser.add_argument("--run-name-prefix", default="qaover")
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--hidden-dims", type=_parse_int_csv, default=[32, 16])
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-prefix", default="overhead_breakdown")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if isinstance(args.batch_size, str) and args.batch_size.lower() != "full":
        args.batch_size = int(args.batch_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: list[dict[str, Any]] = []

    run_index = 0
    for qa_mode in args.qa_samplers:
        for subset_size in args.subset_sizes:
            if int(subset_size) <= 0:
                raise ValueError("All subset_sizes must be positive integers.")

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
                    qa_sampler_mode=qa_mode,
                    subset_size=int(subset_size),
                    step_size=float(args.step_size),
                    num_reads=int(args.num_reads),
                    lr=1e-2,
                    damping=1e-4,
                )

                run_name = (
                    f"{args.run_name_prefix}-{args.model}-{args.dataset}-"
                    f"qa:{qa_mode}:subset{subset_size}-seed{seed}-{run_index:03d}"
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

                component_means = {component: _mean_epoch_metric(epoch_history, component) for component in _COMPONENTS}
                total_time_mean = float(sum(component_means.values()))

                run_rows.append(
                    {
                        "model": args.model,
                        "dataset": args.dataset,
                        "qa_sampler_mode": qa_mode,
                        "subset_size": int(subset_size),
                        "seed": int(seed),
                        "run_id": summary.get("run_id"),
                        "epochs": int(args.epochs),
                        "build_bqm_time_sec_mean": component_means["build_bqm_time_sec"],
                        "transfer_time_sec_mean": component_means["transfer_time_sec"],
                        "sampling_time_sec_mean": component_means["sampling_time_sec"],
                        "update_time_sec_mean": component_means["update_time_sec"],
                        "step_time_components_sum_sec_mean": total_time_mean,
                        "build_bqm_share_pct": _to_percent(component_means["build_bqm_time_sec"], total_time_mean),
                        "transfer_share_pct": _to_percent(component_means["transfer_time_sec"], total_time_mean),
                        "sampling_share_pct": _to_percent(component_means["sampling_time_sec"], total_time_mean),
                        "update_share_pct": _to_percent(component_means["update_time_sec"], total_time_mean),
                    }
                )

                run_index += 1

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in run_rows:
        grouped[(str(row["qa_sampler_mode"]), int(row["subset_size"]))].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (qa_mode, subset_size), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        def _metric(name: str) -> list[float]:
            return [float(row[name]) for row in rows]

        build_share = _metric("build_bqm_share_pct")
        transfer_share = _metric("transfer_share_pct")
        sampling_share = _metric("sampling_share_pct")
        update_share = _metric("update_share_pct")

        summary_rows.append(
            {
                "qa_sampler_mode": qa_mode,
                "subset_size": subset_size,
                "num_runs": len(rows),
                "build_bqm_time_sec_mean": float(mean(_metric("build_bqm_time_sec_mean"))),
                "transfer_time_sec_mean": float(mean(_metric("transfer_time_sec_mean"))),
                "sampling_time_sec_mean": float(mean(_metric("sampling_time_sec_mean"))),
                "update_time_sec_mean": float(mean(_metric("update_time_sec_mean"))),
                "step_time_components_sum_sec_mean": float(mean(_metric("step_time_components_sum_sec_mean"))),
                "build_bqm_share_pct_mean": float(mean(build_share)),
                "build_bqm_share_pct_std": float(stdev(build_share)) if len(build_share) > 1 else 0.0,
                "transfer_share_pct_mean": float(mean(transfer_share)),
                "transfer_share_pct_std": float(stdev(transfer_share)) if len(transfer_share) > 1 else 0.0,
                "sampling_share_pct_mean": float(mean(sampling_share)),
                "sampling_share_pct_std": float(stdev(sampling_share)) if len(sampling_share) > 1 else 0.0,
                "update_share_pct_mean": float(mean(update_share)),
                "update_share_pct_std": float(stdev(update_share)) if len(update_share) > 1 else 0.0,
            }
        )

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.output_prefix}_{timestamp}"

    run_json = output_dir / f"{stem}_runs.json"
    run_csv = output_dir / f"{stem}_runs.csv"
    summary_json = output_dir / f"{stem}_summary.json"
    summary_csv = output_dir / f"{stem}_summary.csv"

    _write_json(run_json, run_rows)
    _write_csv(run_csv, run_rows)
    _write_json(summary_json, summary_rows)
    _write_csv(summary_csv, summary_rows)

    plot_paths: list[Path] = []
    by_sampler: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        by_sampler[str(row["qa_sampler_mode"])].append(row)

    for qa_mode, rows in by_sampler.items():
        safe_mode = qa_mode.replace("/", "_").replace(":", "_")
        plot_path = output_dir / f"{stem}_{safe_mode}_share_vs_subset_size.png"
        _plot_sampler_breakdown(qa_mode, rows, plot_path)
        plot_paths.append(plot_path)

    print(
        "Saved overhead breakdown artifacts:\n"
        f"- Runs: {run_json} and {run_csv}\n"
        f"- Summary: {summary_json} and {summary_csv}\n"
        "- Plots:\n"
        + "\n".join(f"  - {path}" for path in plot_paths)
    )


if __name__ == "__main__":
    main()
