#!/usr/bin/env python3
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from src.losses import RidgeLoss, SVMSquaredHingeLoss
from src.models import Logistic, QuadraticMLP, Ridge, SVM
from src.newton_optimizer import NewtonOptimizer
from src.quadratic_annealing_optimizer import QuadraticAnnealingOptimizer
from src.training import train
from src.utils import build_sampler, data_load_and_prep, set_global_seed


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_float_csv(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def _infer_dimensions(train_loader: torch.utils.data.DataLoader) -> tuple[int, int]:
    features, targets = train_loader.dataset.tensors
    input_dim = int(features.shape[1])
    output_dim = int(torch.unique(targets).numel())
    return input_dim, output_dim


def _build_model_and_loss(
    model_name: str,
    dataset_name: str,
    train_loader: torch.utils.data.DataLoader,
    hidden_dims: list[int],
) -> tuple[nn.Module, nn.Module]:
    input_dim, output_dim = _infer_dimensions(train_loader)
    normalized_model = model_name.lower()
    normalized_dataset = dataset_name.lower()

    if normalized_model == "ridge":
        if normalized_dataset != "diabetes":
            raise ValueError("Ridge model is supported only for diabetes regression in this script.")
        return Ridge(input_dim=input_dim, output_dim=1), RidgeLoss(alpha=1.0)

    if normalized_model == "svm":
        if normalized_dataset != "breast_cancer":
            raise ValueError("SVM model is supported only for breast_cancer dataset in this script.")
        return SVM(input_dim=input_dim, output_dim=1), SVMSquaredHingeLoss(C=1.0)

    if normalized_model == "logistic":
        if normalized_dataset in {"diabetes", "breast_cancer"}:
            raise ValueError("Logistic model in this script supports multiclass datasets only.")
        return Logistic(input_dim=input_dim, output_dim=output_dim), nn.CrossEntropyLoss()

    if normalized_model == "mlp":
        if normalized_dataset == "diabetes":
            return QuadraticMLP(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=1), nn.MSELoss()
        return QuadraticMLP(input_dim=input_dim, hidden_dim=hidden_dims, output_dim=output_dim), nn.CrossEntropyLoss()

    raise ValueError("model must be one of: mlp, logistic, svm, ridge")


def _build_optimizer(
    optimizer_name: str,
    model: nn.Module,
    sampler_mode: str,
    subset_size: int,
    step_size: float,
    num_reads: int,
    lr: float,
    damping: float,
) -> Any:
    normalized = optimizer_name.lower()
    if normalized == "qa":
        sampler = build_sampler(mode=sampler_mode)
        return QuadraticAnnealingOptimizer(
            sampler=sampler,
            model=model,
            subset_size=subset_size,
            step_size=step_size,
            num_reads=num_reads,
        )
    if normalized == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if normalized == "lbfgs":
        return torch.optim.LBFGS(model.parameters(), lr=lr)
    if normalized == "newton":
        return NewtonOptimizer(model.parameters(), lr=lr, damping=damping)

    raise ValueError("optimizer must be one of: qa, adam, lbfgs, newton")


def _generate_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    samplers = args.samplers if args.optimizer == "qa" else ["not_applicable"]
    subset_sizes = args.subset_sizes if args.optimizer == "qa" else [0]
    step_sizes = args.step_sizes if args.optimizer == "qa" else [0.0]
    num_reads_values = args.num_reads if args.optimizer == "qa" else [0]

    for seed in args.seeds:
        for sampler_mode in samplers:
            for subset_size in subset_sizes:
                for step_size in step_sizes:
                    for num_reads in num_reads_values:
                        configs.append(
                            {
                                "seed": seed,
                                "sampler_mode": sampler_mode,
                                "subset_size": subset_size,
                                "step_size": step_size,
                                "num_reads": num_reads,
                            }
                        )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiment grid and export summaries to JSON/CSV + MLflow.")
    parser.add_argument("--model", required=True, choices=["mlp", "logistic", "svm", "ridge"])
    parser.add_argument("--dataset", required=True, choices=["iris", "wine", "breast_cancer", "digits", "diabetes"])
    parser.add_argument("--optimizer", default="qa", choices=["qa", "adam", "lbfgs", "newton"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", default="32")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--tracking-uri", default=None)
    parser.add_argument("--experiment-name", default="second-order-opt-grid")
    parser.add_argument("--run-name-prefix", default="grid")
    parser.add_argument("--seeds", type=_parse_int_csv, default=[42])
    parser.add_argument("--samplers", type=_parse_csv, default=["simulated"])
    parser.add_argument("--subset-sizes", type=_parse_int_csv, default=[12])
    parser.add_argument("--step-sizes", type=_parse_float_csv, default=[0.05])
    parser.add_argument("--num-reads", type=_parse_int_csv, default=[100])
    parser.add_argument("--hidden-dims", type=_parse_int_csv, default=[32, 16])
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--output-prefix", default="grid_summary")
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if isinstance(args.batch_size, str) and args.batch_size.lower() != "full":
        args.batch_size = int(args.batch_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = _generate_configs(args)
    if args.max_runs > 0:
        configs = configs[: args.max_runs]

    results: list[dict[str, Any]] = []

    for run_idx, config in enumerate(configs):
        seed = int(config["seed"])
        set_global_seed(seed)

        train_loader, test_loader = data_load_and_prep(
            dataset_name=args.dataset,
            test_size=args.test_size,
            random_state=seed,
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
            optimizer_name=args.optimizer,
            model=model,
            sampler_mode=config["sampler_mode"],
            subset_size=int(config["subset_size"]),
            step_size=float(config["step_size"]),
            num_reads=int(config["num_reads"]),
            lr=args.lr,
            damping=args.damping,
        )

        run_name = f"{args.run_name_prefix}-{args.model}-{args.dataset}-{run_idx:03d}"
        summary = train(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            loss_fn=loss_fn,
            c_device=args.device,
            optimizer=optimizer,
            epochs=args.epochs,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment_name,
            log_batch_metrics=True,
            run_name=run_name,
            verbose=not args.quiet,
            seed=seed,
        )

        record = {
            "model": args.model,
            "dataset": args.dataset,
            "optimizer": args.optimizer,
            "sampler_mode": config["sampler_mode"],
            "subset_size": int(config["subset_size"]),
            "step_size": float(config["step_size"]),
            "num_reads": int(config["num_reads"]),
            "seed": seed,
            **summary,
        }
        results.append(record)

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

    print(f"Saved {len(results)} run summaries to {json_path} and {csv_path}")


if __name__ == "__main__":
    main()
