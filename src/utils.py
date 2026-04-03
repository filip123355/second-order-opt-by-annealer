import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from torch.utils.data import DataLoader
from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
from dimod import ExactSolver
from dwave.samplers import SimulatedAnnealingSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from src.gpu_simulated_annealing.gpu_simulated_annealing import GPUSimulatedAnnealingSampler
from .losses import RidgeLoss
from typing import Optional

def build_sampler(mode: str = "simulated",
) -> DWaveSampler | ExactSolver | SimulatedAnnealingSampler | GPUSimulatedAnnealingSampler:
    normalized_mode = mode.lower() 

    # TODO: Discuss adding the controlls with Paweł

    if normalized_mode == "dwave":
        pass

    if normalized_mode == "dwave":
        try:
            token = os.environ.get("DWAVE_API_TOKEN")
            endpoint = os.environ.get("DWAVE_API_ENDPOINT")
            return EmbeddingComposite(DWaveSampler(token=token, endpoint=endpoint))
        except Exception as exc:
            print(f"Falling back to simulated annealing because the QPU is unavailable: {exc}")
            return SimulatedAnnealingSampler()
        
    elif normalized_mode == "hybrid":
        try:
            token = os.environ.get("DWAVE_API_TOKEN")
            endpoint = os.environ.get("DWAVE_API_ENDPOINT")
            return LeapHybridSampler(token=token, endpoint=endpoint)
        except Exception as exc:
            print(f"Falling back to simulated annealing because the hybrid solver is unavailable: {exc}")
            return SimulatedAnnealingSampler()

    elif normalized_mode == "exact":
        return ExactSolver()

    elif normalized_mode == "simulated":
        return SimulatedAnnealingSampler()
    
    elif normalized_mode == "gpu_simulated":
        return GPUSimulatedAnnealingSampler()

    raise ValueError("mode must be one of: simulated, exact, dwave")


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_ev = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            if isinstance(loss_fn, RidgeLoss):
                loss = loss_fn(logits, targets, model)
                targets_for_mse = targets.view_as(logits) if targets.shape != logits.shape else targets
                batch_mse = F.mse_loss(logits, targets_for_mse, reduction="mean")
                total_ev += float(batch_mse.item()) * targets.size(0)
            else:
                loss = loss_fn(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += int(targets.size(0))

    average_loss = total_loss / total_examples

    if isinstance(loss_fn, RidgeLoss):
        average_mse = total_ev / total_examples
        return average_loss, average_mse
    accuracy = total_correct / total_examples

    return average_loss, accuracy


def data_load_and_prep(dataset_name: str, 
                       test_size: float = 0.3,
                       random_state: int|None = 42,
                       batch_size: int|str = 32,
                       shuffle: bool = True,
                       ) -> tuple[DataLoader, DataLoader]:
    """Load and preprocess the dataset."""
    
    if dataset_name.lower() == "iris":
        from sklearn.datasets import load_iris
        dataset = load_iris()
    elif dataset_name.lower() == "wine":
        from sklearn.datasets import load_wine
        dataset = load_wine()
    elif dataset_name.lower() == "breast_cancer":
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
    elif dataset_name.lower() == "digits":
        from sklearn.datasets import load_digits
        dataset = load_digits()
    elif dataset_name.lower() == "diabetes":
        from sklearn.datasets import load_diabetes
        dataset = load_diabetes()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: iris, wine, breast_cancer.")


    X, y = dataset.data, dataset.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X = torch.FloatTensor(X)
    if dataset_name.lower() == "diabetes":
        y = torch.FloatTensor(y)
    else:
        y = torch.LongTensor(y)
    
    if not random_state:
        random_state = torch.seed()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None, shuffle=shuffle,
    )
    
    if isinstance(batch_size, str) and batch_size.lower() == "full":
        batch_size = len(X_train)

    training_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader