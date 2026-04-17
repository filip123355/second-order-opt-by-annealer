#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.training import train
from src.utils import data_load_and_prep, set_global_seed
from src.script_utils import (
    _parse_csv,
    _parse_int_csv,
    _build_model_and_loss,
    _build_optimizer,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multiple optimizers on a single task (same model + dataset)."
    )
    parser.add_argument("--model", required=True, choices=["mlp", "logistic", "svm", "ridge"])
    parser.add_argument("--dataset", required=True, choices=["iris", "wine", "breast_cancer", "digits", "diabetes"])
    parser.add_argument("--optimizers", type=_parse_csv, default=["qa", "adam", "lbfgs", "newton"])
    parser.add_argument("--qa-samplers", type=_parse_csv, default=["simulated"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", default="32")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default="optimizer-comparison")
    parser.add_argument("--run-name-prefix", default="compare")
    parser.add_argument("--subset-size", type=int, default=12)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--num-reads", type=int, default=100)
    parser.add_argument("--hidden-dims", type=_parse_int_csv, default=[32, 16])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-prefix", default="optimizer_comparison")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if isinstance(args.batch_size, str) and args.batch_size.lower() != "full":
        args.batch_size = int(args.batch_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_global_seed(int(args.seed))

    train_loader, test_loader = data_load_and_prep(
        dataset_name=args.dataset,
        test_size=args.test_size,
        random_state=args.seed,
        batch_size=args.batch_size,
        shuffle=True,
    )

    results: list[dict[str, Any]] = []
    run_index = 0

    for optimizer_name in args.optimizers:
        normalized_optimizer = optimizer_name.lower()
        qa_modes = args.qa_samplers if normalized_optimizer == "qa" else ["not_applicable"]

        for qa_mode in qa_modes:
            set_global_seed(int(args.seed))
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
                subset_size=args.subset_size,
                step_size=args.step_size,
                num_reads=args.num_reads,
                lr=args.lr,
                damping=args.damping,
            )

            run_name = (
                f"{args.run_name_prefix}-{args.model}-{args.dataset}-"
                f"{normalized_optimizer}-{qa_mode}-{run_index:03d}"
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
                seed=int(args.seed),
            )

            record = {
                "model": args.model,
                "dataset": args.dataset,
                "optimizer": normalized_optimizer,
                "qa_sampler_mode": qa_mode,
                "seed": int(args.seed),
                "subset_size": int(args.subset_size) if normalized_optimizer == "qa" else None,
                "step_size": float(args.step_size) if normalized_optimizer == "qa" else None,
                "num_reads": int(args.num_reads) if normalized_optimizer == "qa" else None,
                **summary,
            }
            results.append(record)
            run_index += 1

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.output_prefix}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    if results:
        fieldnames = list(results[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print(f"Saved optimizer comparison ({len(results)} runs) to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
