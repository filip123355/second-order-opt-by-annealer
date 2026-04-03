# Second-order annealer experiments

This repository demonstrates a hybrid second-order optimization approach: we form a local
quadratic approximation of the loss around the current parameters, convert that approximation
to a Binary Quadratic Model (BQM), and optimize the BQM using an annealer or classical sampler.
The codebase contains a small PyTorch model suite, the quadratic-annealing optimizer, utility
scripts, and example notebooks used to reproduce experiments.

## Highlights

- Local quadratic approximation + annealer-based discrete proposals
- Support for exact solver and simulated annealing backends
- Selective-parameter updates: operate on a block of parameters per step
- Pluggable loss functions (examples include ridge-regularized losses)

## Quick start

1. Create and activate a Python virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

1. Run a notebook example (open in your browser):

```bash
jupyter notebook notebooks/experiments.ipynb
```

1. Example: run a training script or reproduce a notebook cell that instantiates the optimizer:

```python
from src.quadratic_annealing_optimizer import QuadraticAnnealingOptimizer
from dimod import ExactSolver

optimizer = QuadraticAnnealingOptimizer(
    sampler=ExactSolver(),
    model=my_model,
    subset_size=12,
    step_size=0.05,
    num_reads=200,
    beta1=0.9,  # Adam-style first moment; `beta` is still accepted as alias
    beta2=0.999,
    eps=1e-8,
)
```

## Where useful experiment artifacts are stored

- `mlruns/`: MLflow run directories (if experiments used MLflow)
- `data/`: input datasets (e.g., MNIST raw files)
- `notebooks/`: reproducible example notebooks

## Development notes

- Python compatibility: developed against Python 3.10+. Use a virtual environment.
- Tests and CI: no formal test suite is included; run notebooks to verify experiments.
- If you modify the optimizer state logic, ensure acceptance/rejection semantics are
  preserved: proposed moment/state updates should normally only be committed after an
  accepted candidate.
