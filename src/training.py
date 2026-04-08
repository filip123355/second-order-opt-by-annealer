import torch
import torch.nn as nn
import mlflow
import os

from time import perf_counter
from torch.utils.data import DataLoader
from typing import Optional, Any

from .quadratic_annealing_optimizer import QuadraticAnnealingOptimizer
from .models import QuadraticMLP
from .newton_optimizer import NewtonOptimizer
from .utils import evaluate
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
):
    device = torch.device(c_device)
    model.to(device)

    default_tracking_uri = f"file:{os.path.abspath('../mlruns')}"
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

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "optimizer": type(optimizer).__name__,
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "device": str(device),
            }
        )

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

        for epoch in range(epochs):
            model.train()
            batch_energies = []
            accepted_steps = 0

            for features, targets in train_loader:
                features = features.to(device)
                targets = targets.to(device)
                
                if isinstance(optimizer, QuadraticAnnealingOptimizer):
                    step_info = optimizer.step(features, targets, loss_fn)
                    batch_energies.append(step_info["quadratic_energy"])
                    accepted_steps += int(step_info["accepted"])

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

                if log_batch_metrics and isinstance(optimizer, QuadraticAnnealingOptimizer):
                    log = {"batch_loss": float(step_info["loss"])}
                    log["batch_quadratic_energy"] = float(step_info["quadratic_energy"])
                    log["batch_accepted"] = float(step_info["accepted"])
                    mlflow.log_metrics(
                        log,
                        step=global_step,
                    )
                global_step += 1

            train_loss, train_metric = evaluate(model, train_loader, loss_fn, device)
            test_loss, test_metric = evaluate(model, test_loader, loss_fn, device)

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

            if isinstance(optimizer, QuadraticAnnealingOptimizer):
                acceptance_rate = accepted_steps / len(train_loader)
                log["train_quadratic_energy"] = float(sum(batch_energies) / len(batch_energies))
                log["acceptance_rate"] = float(acceptance_rate)

            mlflow.log_metrics(
                log,
                step=epoch,
            )

            if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"train_{"mse" if isinstance(loss_fn, RidgeLoss) else "acc"}={train_metric:.3f} | "
                    f"test_loss={test_loss:.4f} | "
                    f"test_{"mse" if isinstance(loss_fn, RidgeLoss) else "acc"}={test_metric:.3f} | "
                )

        training_time = perf_counter() - start_time
        mlflow.log_metric("training_time_sec", float(training_time))
        mlflow.pytorch.log_model(model, name="model")