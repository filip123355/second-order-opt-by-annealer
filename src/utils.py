import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import ExactSolver
from dwave.samplers import SimulatedAnnealingSampler


def build_sampler(mode: str = "simulated"):
    normalized_mode = mode.lower()

    if normalized_mode == "dwave":
        try:
            return EmbeddingComposite(DWaveSampler()), "dwave"
        except Exception as exc:
            print(f"Falling back to simulated annealing because the QPU is unavailable: {exc}")
            return SimulatedAnnealingSampler(), "simulated"

    if normalized_mode == "exact":
        return ExactSolver(), "exact"

    if normalized_mode == "simulated":
        return SimulatedAnnealingSampler(), "simulated"
    
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