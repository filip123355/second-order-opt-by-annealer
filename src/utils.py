import torch
import torch.nn as nn
import os

from torch.utils.data import DataLoader
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import ExactSolver
from dwave.samplers import SimulatedAnnealingSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset

def build_sampler(mode: str = "simulated"):
    normalized_mode = mode.lower() 

    if normalized_mode == "dwave":
        try:
            token = os.environ.get("DWAVE_API_TOKEN")
            endpoint = os.environ.get("DWAVE_API_ENDPOINT")
            return EmbeddingComposite(DWaveSampler(token=token, endpoint=endpoint))
        except Exception as exc:
            print(f"Falling back to simulated annealing because the QPU is unavailable: {exc}")
            return SimulatedAnnealingSampler()

    if normalized_mode == "exact":
        return ExactSolver()

    if normalized_mode == "simulated":
        return SimulatedAnnealingSampler()
    
    # if normalized_mode == "gpu_simpulated":
    #     return None

    raise ValueError("mode must be one of: simulated, exact, dwave")


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    ):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            total_loss += float(loss_fn(logits, targets).item()) * targets.size(0)
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += int(targets.size(0))

    average_loss = total_loss / total_examples
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
    y = torch.LongTensor(y)
    
    if not random_state:
        random_state = torch.seed()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y, shuffle=shuffle,
    )
    
    if isinstance(batch_size, str) and batch_size.lower() == "full":
        batch_size = len(X_train)

    training_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader