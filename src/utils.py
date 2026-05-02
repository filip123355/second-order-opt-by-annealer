import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np

from torch.utils.data import DataLoader
from dwave.system import DWaveSampler, LazyFixedEmbeddingComposite
from dimod import ExactSolver
from dwave.samplers import SimulatedAnnealingSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from src.gpu_simulated_annealing.gpu_simulated_annealing import GPUSimulatedAnnealingSampler
from .losses import RidgeLoss, SVMSquaredHingeLoss
from typing import Optional, Any
from dotenv import load_dotenv


def set_global_seed(seed: int) -> None:
    """Set global random seed across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_sampler(mode: str = "simulated",
) -> Any:
    normalized_mode = mode.lower() 
    
    if normalized_mode == "dwave":
        load_dotenv()
        try:
            token = os.environ.get("DWAVE_API_TOKEN")
            endpoint = os.environ.get("DWAVE_API_ENDPOINT")
            return LazyFixedEmbeddingComposite(DWaveSampler(token=token, endpoint=endpoint)) # Resuing the same embedding across all runs
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize D-Wave sampler. Please check your API token and endpoint: {exc}")
        
    elif normalized_mode == "hybrid":
        # try:
        #     token = os.environ.get("DWAVE_API_TOKEN")
        #     endpoint = os.environ.get("DWAVE_API_ENDPOINT")
        #     return LeapHybridSampler(token=token, endpoint=endpoint)
        # except Exception as exc:
        #     print(f"Falling back to simulated annealing because the hybrid solver is unavailable: {exc}")
        #     return SimulatedAnnealingSampler()
        raise NotImplementedError("The LeapHybrid Sampler is too resource-intensive. If you are rich, go for it, but it's not suitable for this project.")

    elif normalized_mode == "exact":
        return ExactSolver()

    elif normalized_mode == "simulated":
        return SimulatedAnnealingSampler()
    
    elif normalized_mode == "gpu_simulated":
        return GPUSimulatedAnnealingSampler()

    elif normalized_mode == "veloxq":
        # from .veloxq_sampler import VeloxQSampler
        from veloxq_sdk  import VeloxQSolver, PLGridGH200
        from veloxq_sdk.config import VeloxQAPIConfig, load_config

        load_dotenv() 
        load_config() 
        
        backend = PLGridGH200()
        return VeloxQSolver(backend=backend)
    
    # elif normalized_mode == "veloxq_sbm":
    #     from veloxq_sdk import SBMSolver, PLGridGH200
    #     from veloxq_sdk.config import VeloxQAPIConfig, load_config

    #     load_dotenv() 
    #     load_config() 
        
    #     backend = PLGridGH200()
    #     return SBMSolver(backend=backend)
    # We do not have access to the SMB solver through PLGrid.

    raise ValueError(
        "mode must be one of: simulated, exact, dwave, hybrid, gpu_simulated, "
        "veloxq, veloxq_h100_1, veloxq_h100_2, veloxq_plgrid_gh200, "
        "veloxq_sbm, veloxq_sbm_h100_1, veloxq_sbm_h100_2, veloxq_sbm_plgrid_gh200"
    )


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
            if isinstance(loss_fn, RidgeLoss) or isinstance(loss_fn, SVMSquaredHingeLoss):
                loss = loss_fn(logits, targets, model)
                targets_for_mse = targets.view_as(logits) if targets.shape != logits.shape else targets
                batch_mse = F.mse_loss(logits, targets_for_mse, reduction="mean")
                total_ev += float(batch_mse.item()) * targets.size(0)
            else:
                loss = loss_fn(logits, targets)
            total_loss += float(loss.item()) * targets.size(0)

            if isinstance(loss_fn, SVMSquaredHingeLoss):
                # Binary SVM outputs a single margin per sample; decision boundary is at 0.
                predicted = torch.where(logits.view(-1) >= 0, 1.0, -1.0)
                target_eval = targets.view(-1).float()
                total_correct += int((predicted == target_eval).sum().item())
            elif logits.ndim == 2 and logits.shape[1] == 1:
                # Generic binary case with labels in {0, 1}.
                predicted = (logits.view(-1) >= 0).long()
                target_eval = targets.view(-1).long()
                total_correct += int((predicted == target_eval).sum().item())
            else:
                predicted = logits.argmax(dim=1)
                target_eval = targets.view(-1)
                total_correct += int((predicted == target_eval).sum().item())

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
    elif dataset_name.lower() == "breast_cancer":
        y *= 2
        y -= 1
        y = torch.FloatTensor(y)
    else:
        y = torch.LongTensor(y)
    
    if not random_state:
        random_state = torch.seed()
    
    stratify_targets = None if dataset_name.lower() == "diabetes" else y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_targets,
        shuffle=shuffle,
    )
    
    if isinstance(batch_size, str) and batch_size.lower() == "full":
        batch_size = len(X_train)

    training_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader