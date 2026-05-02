import torch
import torch.nn as nn
import mlflow
import os
from pathlib import Path

from time import perf_counter
from torch.utils.data import DataLoader
from typing import Optional, Any

from .quadratic_annealing_optimizer import QuadraticAnnealingOptimizer
from .models import QuadraticMLP
from .newton_optimizer import NewtonOptimizer
from .utils import evaluate, set_global_seed
from .losses import RidgeLoss, SVMSquaredHingeLoss


def train(
    model: QuadraticMLP,
    train_loader: DataLoader,
    test_loader: DataLoader,
    loss_fn: nn.Module,
    c_device: str,
    optimizer: Any,
    epochs: int,
    tracking_uri: str | None = None,
    experiment_name: str = "second-order-opt-by-annealer",
    log_batch_metrics: bool = True,
    run_name: str | None = None,
    verbose: bool = True,
    seed: int | None = None,
    collect_epoch_history: bool = False,
    zero_acceptance_patience: int = 6,
    warm_start_epochs: int = 1,
) -> dict[str, object]:
    if seed is not None:
        set_global_seed(seed)

    warm_start_epochs = max(0, int(warm_start_epochs))

    device = torch.device(c_device)
    model.to(device)

    def _sync_if_cuda() -> None:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)

    default_tracking_uri = Path(__file__).resolve().parents[1].joinpath("mlruns").as_uri()
    requested_tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or default_tracking_uri
    mlflow.set_tracking_uri(requested_tracking_uri)
    
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        if requested_tracking_uri == default_tracking_uri:
            raise
        print(
            f"MLflow tracking URI '{requested_tracking_uri}' is unavailable ({exc}). "
            f"Falling back to local store at {default_tracking_uri}."
        )
        mlflow.set_tracking_uri(default_tracking_uri)
        mlflow.set_experiment(experiment_name)

    summary: dict[str, object] = {}
    epoch_history: list[dict[str, float | int]] = []
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(
            {
                "optimizer": type(optimizer).__name__,
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "device": str(device),
                "experiment_name": experiment_name,
                "tracking_uri": requested_tracking_uri,
                "warm_start_epochs": warm_start_epochs,
            }
        )
        if seed is not None:
            mlflow.log_param("seed", int(seed))
        mlflow.log_param("experiment_id", str(active_run.info.experiment_id))

        model_config = {
            name: module 
            for name, module 
            in model.named_modules()
        }
            
        mlflow.log_dict(model_config, "configs/model.json")

        opt_defaults = optimizer.defaults
        opt_config = {key: optimizer.param_groups[0][key] for key, value in opt_defaults.items()}
        mlflow.log_dict(opt_config, "configs/optimizer.json")

        start_time = perf_counter()
        global_step = 0
        raw_total_optimization_time_sec = 0.0
        total_optimization_time_sec = 0.0
        total_train_eval_time_sec = 0.0
        total_test_eval_time_sec = 0.0
        sampler_name_logged = False
        solver_id_logged = False
        zero_acceptance_streak = 0
        early_stopped = False
        early_stop_reason: str | None = None
        completed_epochs = 0

        for epoch in range(epochs):
            epoch_start = perf_counter()
            log_this_epoch = epoch >= warm_start_epochs
            model.train()
            batch_energies = []
            accepted_steps = 0
            epoch_optimizer_time_values: list[float] = []
            backend_metric_values: dict[str, list[float]] = {
                "qpu_access_time": [],
                "qpu_sampling_time": [],
                "qpu_readout_time": [],
                "qpu_anneal_time_per_sample": [],
                "qpu_delay_time_per_sample": [],
                "chain_strength": [],
                "chain_break_fraction": [],
                "avg_chain_length": [],
                "max_chain_length": [],
                "min_chain_length": [],
                "num_chains": [],
                "num_physical_qubits": [],
                "problem_construction_time_sec": [],
                "transfer_time_sec": [],
                "sampling_time_sec": [],
                "sample_total_time_sec": [],
                "update_time_sec": [],
            }

            for features, targets in train_loader:
                
                _sync_if_cuda()
                step_start = perf_counter()
                features = features.to(device)
                targets = targets.to(device)
                
                if isinstance(optimizer, QuadraticAnnealingOptimizer):
                    step_info = optimizer.step(features, targets, loss_fn)
                    _sync_if_cuda()
                    step_wall_time_sec = float(perf_counter() - step_start)
                    batch_energies.append(step_info["quadratic_energy"])
                    accepted_steps += int(step_info["accepted"])
                    step_optimization_time_sec = float(step_info.get("optimization_time_sec", step_wall_time_sec))

                    if not sampler_name_logged and step_info.get("sampler_name"):
                        mlflow.log_param("sampler_name", str(step_info["sampler_name"]))
                        sampler_name_logged = True
                    if not solver_id_logged and step_info.get("solver_id"):
                        mlflow.log_param("solver_id", str(step_info["solver_id"]))
                        solver_id_logged = True

                    for metric_name in backend_metric_values:
                        metric_value = step_info.get(metric_name)
                        if metric_value is None:
                            continue
                        cast_value = float(metric_value)
                        backend_metric_values[metric_name].append(cast_value)

                else:
                    if isinstance(optimizer, NewtonOptimizer):
                        def closure():
                            optimizer.zero_grad(set_to_none=True)
                            logits = model(features)
                            if isinstance(loss_fn, RidgeLoss) or isinstance(loss_fn, SVMSquaredHingeLoss):
                                loss = loss_fn(logits, targets, model)
                            else:
                                loss = loss_fn(logits, targets)
                            return loss
                        optimizer.step(closure)
                    else:
                        def closure():
                            optimizer.zero_grad(set_to_none=True)
                            logits = model(features)
                            if isinstance(loss_fn, RidgeLoss) or isinstance(loss_fn, SVMSquaredHingeLoss):
                                loss = loss_fn(logits, targets, model)
                            else:
                                loss = loss_fn(logits, targets)
                            loss.backward()
                            return loss
                        optimizer.step(closure)
                    _sync_if_cuda()
                    step_wall_time_sec = float(perf_counter() - step_start)
                    step_optimization_time_sec = step_wall_time_sec
                epoch_optimizer_time_values.append(step_optimization_time_sec)

                if log_batch_metrics and log_this_epoch:
                    log = {
                        "optimization_time_sec": step_optimization_time_sec,
                    }

                    if isinstance(optimizer, QuadraticAnnealingOptimizer):
                        log["optimization_wall_time_sec"] = step_wall_time_sec

                    if isinstance(optimizer, QuadraticAnnealingOptimizer):
                        log["batch_loss"] = float(step_info["loss"])
                        log["batch_quadratic_energy"] = float(step_info["quadratic_energy"])
                        log["batch_accepted"] = float(step_info["accepted"])

                        for metric_name in backend_metric_values:
                            metric_value = step_info.get(metric_name)
                            if metric_value is None:
                                continue
                            log[f"batch_{metric_name}"] = float(metric_value)

                    mlflow.log_metrics(
                        log,
                        step=global_step,
                    )

                global_step += 1

            _sync_if_cuda()
            train_eval_start = perf_counter()
            train_loss, train_metric = evaluate(model, train_loader, loss_fn, device)
            _sync_if_cuda()
            train_eval_time_sec = float(perf_counter() - train_eval_start)

            _sync_if_cuda()
            test_eval_start = perf_counter()
            test_loss, test_metric = evaluate(model, test_loader, loss_fn, device)
            _sync_if_cuda()
            test_eval_time_sec = float(perf_counter() - test_eval_start)

            if isinstance(loss_fn, RidgeLoss):
                log = {
                    "train_loss": float(train_loss),
                    "train_mse": float(train_metric),
                    "test_loss": float(test_loss),
                    "test_mse": float(test_metric),
                }
            else:
                log = {
                    "train_loss": float(train_loss),
                    "train_accuracy": float(train_metric),
                    "test_loss": float(test_loss),
                    "test_accuracy": float(test_metric),
                }

            optimization_time_sec = float(sum(epoch_optimizer_time_values))
            raw_total_optimization_time_sec += optimization_time_sec
            total_train_eval_time_sec += train_eval_time_sec
            total_test_eval_time_sec += test_eval_time_sec
            epoch_evaluation_time_sec = float(train_eval_time_sec + test_eval_time_sec)
            epoch_mlflow_logging_time_sec = 0.0
            if log_this_epoch:
                total_optimization_time_sec += optimization_time_sec
                log["optimization_time_sec"] = optimization_time_sec
                log["epoch_train_evaluation_time_sec"] = train_eval_time_sec
                log["epoch_test_evaluation_time_sec"] = test_eval_time_sec
                log["epoch_evaluation_time_sec"] = epoch_evaluation_time_sec

                if isinstance(optimizer, QuadraticAnnealingOptimizer):
                    acceptance_rate = accepted_steps / len(train_loader)
                    if acceptance_rate == 0.0:
                        zero_acceptance_streak += 1
                    else:
                        zero_acceptance_streak = 0

                    log["train_quadratic_energy"] = float(sum(batch_energies) / len(batch_energies))
                    log["acceptance_rate"] = float(acceptance_rate)
                    log["zero_acceptance_streak"] = float(zero_acceptance_streak)

                    for metric_name, values in backend_metric_values.items():
                        if values:
                            log[f"train_{metric_name}"] = float(sum(values) / len(values))

                mlflow_logging_start = perf_counter()
                mlflow.log_metrics(
                    log,
                    step=epoch,
                )
                epoch_mlflow_logging_time_sec = float(perf_counter() - mlflow_logging_start)
            epoch_total_time_sec = float(perf_counter() - epoch_start)

            if collect_epoch_history and log_this_epoch:
                elapsed_time_sec = float(perf_counter() - start_time)
                epoch_entry: dict[str, float | int] = {
                    "epoch": int(epoch),
                    "elapsed_time_sec": elapsed_time_sec,
                    "epoch_total_time_sec": epoch_total_time_sec,
                    "optimization_time_sec": optimization_time_sec,
                    "epoch_train_evaluation_time_sec": train_eval_time_sec,
                    "epoch_test_evaluation_time_sec": test_eval_time_sec,
                    "epoch_evaluation_time_sec": epoch_evaluation_time_sec,
                    "epoch_mlflow_logging_time_sec": epoch_mlflow_logging_time_sec,
                    "epoch_non_optimizer_time_sec": float(epoch_total_time_sec - optimization_time_sec),
                    "train_loss": float(train_loss),
                    "test_loss": float(test_loss),
                }

                if isinstance(loss_fn, RidgeLoss):
                    epoch_entry["train_metric"] = float(train_metric)  # MSE
                    epoch_entry["test_metric"] = float(test_metric)   # MSE
                else:
                    epoch_entry["train_metric"] = float(train_metric)  # Accuracy
                    epoch_entry["test_metric"] = float(test_metric)   # Accuracy

                if isinstance(optimizer, QuadraticAnnealingOptimizer):
                    epoch_entry["acceptance_rate"] = float(accepted_steps / len(train_loader))
                    epoch_entry["zero_acceptance_streak"] = float(zero_acceptance_streak)
                    epoch_entry["train_quadratic_energy"] = float(sum(batch_energies) / len(batch_energies))
                    for metric_name, values in backend_metric_values.items():
                        if values:
                            epoch_entry[metric_name] = float(sum(values) / len(values))

                epoch_history.append(epoch_entry)

            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"train_{'mse' if isinstance(loss_fn, RidgeLoss) else 'acc'}={train_metric:.3f} | "
                    f"test_loss={test_loss:.4f} | "
                    f"test_{'mse' if isinstance(loss_fn, RidgeLoss) else 'acc'}={test_metric:.3f} | "
                )

            completed_epochs = epoch + 1
            if isinstance(optimizer, QuadraticAnnealingOptimizer):
                if zero_acceptance_patience > 0 and zero_acceptance_streak >= zero_acceptance_patience:
                    early_stopped = True
                    early_stop_reason = (
                        f"acceptance_rate_was_zero_for_{zero_acceptance_patience}_consecutive_epochs"
                    )
                    mlflow.log_metric("zero_acceptance_streak", float(zero_acceptance_streak), step=epoch)
                    mlflow.set_tag("early_stop_reason", early_stop_reason)
                    break
        
        if epoch_history:
            sum_optimization_time_sec = sum(
                float(epoch_history[i].get("optimization_time_sec", 0.0)) for i in range(len(epoch_history))
            )
        else:
            sum_optimization_time_sec = float(total_optimization_time_sec)
        _sync_if_cuda()
        training_time = perf_counter() - start_time
        mlflow.log_metric("optimization_time_sec", float(total_optimization_time_sec))
        mlflow.log_metric("raw_optimization_time_sec", float(raw_total_optimization_time_sec))
        mlflow.log_metric("sum_epoch_optimization_time_sec", float(sum_optimization_time_sec))
        mlflow.log_metric("train_evaluation_time_sec", float(total_train_eval_time_sec))
        mlflow.log_metric("test_evaluation_time_sec", float(total_test_eval_time_sec))
        mlflow.log_metric("evaluation_time_sec", float(total_train_eval_time_sec + total_test_eval_time_sec))
        mlflow.log_metric("training_time_sec", float(training_time))
        mlflow.pytorch.log_model(model, name="model")

        summary = {
            "run_id": active_run.info.run_id,
            "experiment_id": str(active_run.info.experiment_id),
            "experiment_name": experiment_name,
            "tracking_uri": requested_tracking_uri,
            "warm_start_epochs": int(warm_start_epochs),
            "optimization_time_sec": float(total_optimization_time_sec),
            "raw_optimization_time_sec": float(raw_total_optimization_time_sec),
            "sum_epoch_optimization_time_sec": float(sum_optimization_time_sec),
            "train_evaluation_time_sec": float(total_train_eval_time_sec),
            "test_evaluation_time_sec": float(total_test_eval_time_sec),
            "evaluation_time_sec": float(total_train_eval_time_sec + total_test_eval_time_sec),
            "training_time_sec": float(training_time),
            "final_train_loss": float(train_loss),
            "final_test_loss": float(test_loss),
            "final_train_metric": float(train_metric),
            "final_test_metric": float(test_metric),
            "optimizer": type(optimizer).__name__,
            "seed": int(seed) if seed is not None else None,
            "epochs_completed": int(completed_epochs),
            "early_stopped": early_stopped,
            "early_stop_reason": early_stop_reason,
            "zero_acceptance_patience": int(zero_acceptance_patience),
            "final_zero_acceptance_streak": int(zero_acceptance_streak),
        }
        if collect_epoch_history:
            summary["epoch_history"] = epoch_history

    return summary
