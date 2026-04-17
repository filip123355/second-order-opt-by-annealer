import torch
import torch.nn as nn 

from .models import QuadraticMLP, Logistic, SVM, Ridge
from .losses import RidgeLoss, SVMSquaredHingeLoss
from .newton_optimizer import NewtonOptimizer
from .quadratic_annealing_optimizer import QuadraticAnnealingOptimizer
from .utils import build_sampler
from typing import Any


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
    qa_sampler_mode: str,
    subset_size: int,
    step_size: float,
    num_reads: int,
    lr: float,
    damping: float,
) -> Any:
    normalized = optimizer_name.lower()
    if normalized == "qa":
        sampler = build_sampler(mode=qa_sampler_mode)
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