# Second-order annealer experiments

This repository demonstrates a hybrid second-order optimization approach: we form a local
quadratic approximation of the loss around the current parameters, convert that approximation
to a Binary Quadratic Model (BQM), and optimize the BQM using an annealer or classical sampler.
The codebase contains a small PyTorch model suite, the quadratic-annealing optimizer, utility
scripts, and example notebooks used to reproduce experiments.

## Highlights

- Local quadratic approximation + annealer-based discrete proposals
- Support for exact, simulated, hybrid, D-Wave, GPU simulated annealing and veloxQ backends
- Selective-parameter updates: operate on a block of parameters per step
- Pluggable loss functions (examples include ridge-regularized losses)
- Batch and epoch logging for annealer telemetry, including QPU timing and qubit temperature when available

## Quick start

Create and activate a Python virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Example: run a training script or reproduce a notebook cell that instantiates the optimizer:

```python
from src.quadratic_annealing_optimizer import QuadraticAnnealingOptimizer
from dimod import ExactSolver

optimizer = QuadraticAnnealingOptimizer(
    sampler=ExactSolver(),
    model=my_model,
    subset_size=12,
    step_size=0.05,
    num_reads=200,
    beta=0.9,
)
```

## Quantum backend configuration

For the full access to D-Wave's annealers or VeloxQ algorithms you need a D-Wave and VeloxQ API tokens, respecively. Create a `.env` file in the project root with the following content:

```bash
D_WAVE_API_TOKEN=your_dwave_api_token_here
VELOXQ_API_TOKEN=your_veloxq_api_token_here
```

## Reproducible runs

Use a fixed seed for deterministic splits and stochastic operators:

```python
from src.utils import set_global_seed

set_global_seed(42)
```

The `train(...)` function now accepts a `seed` argument and logs it to MLflow.

## Run Experiment Grid (JSON/CSV + MLflow)

Use the grid runner to execute multiple configurations from CLI:

```bash
python scripts/run_experiment_grid.py \
  --model mlp \
  --dataset iris \
  --optimizer qa \
  --samplers simulated,dwave,hybrid \
  --subset-sizes 8,12 \
  --step-sizes 0.03,0.05 \
  --num-reads 100,200 \
  --seeds 7,21,42 \
  --epochs 20 \
  --output-dir results
```

The script saves per-run summaries to JSON/CSV and logs full metrics to MLflow.

## veloxQ backend

The optimizer can use veloxQ through sampler modes passed to `build_sampler(mode=...)`.

Supported veloxQ modes:

- `veloxq` (default VeloxQ solver on H100_1)
- `veloxq_h100_1`
- `veloxq_h100_2`
- `veloxq_plgrid_gh200`
- `veloxq_sbm`
- `veloxq_sbm_h100_1`
- `veloxq_sbm_h100_2`
- `veloxq_sbm_plgrid_gh200`

Example run:

```bash
python scripts/run_experiment_grid.py \
  --model mlp \
  --dataset digits \
  --optimizer qa \
  --samplers veloxq_h100_1 \
  --num-reads 1024 \
  --epochs 10
```

## Compare Optimizers On One Task

Use this script to compare different optimizers on the same model and dataset (workflow similar to notebooks):

```bash
python scripts/run_optimizer_comparison.py \
  --model mlp \
  --dataset iris \
  --optimizers qa,adam,lbfgs,newton \
  --qa-samplers simulated \
  --epochs 20 \
  --seed 42 \
  --output-dir results
```

## Quality Vs Wall-Clock Benchmark

Use this script to compare optimization quality as a function of elapsed wall-clock time.
It exports three levels of artifacts:

- run summaries (`*_runs.json`, `*_runs.csv`)
- per-epoch time series (`*_timeline.json`, `*_timeline.csv`)
- quality at fixed time budgets (`*_budget_raw.*` and aggregated `*_budget_summary.*`)

Example run:

```bash
python scripts/run_quality_vs_wallclock.py \
  --model mlp \
  --dataset digits \
  --optimizers qa,adam,lbfgs,newton \
  --qa-samplers simulated,dwave,hybrid \
  --seeds 7,21,42 \
  --epochs 30 \
  --quality-metric test_metric \
  --quality-direction auto \
  --time-grid-points 20 \
  --output-dir results
```

Tip:

- For classification datasets, `test_metric` means accuracy and should be maximized.
- For diabetes regression, `test_metric` means MSE and should be minimized.
- `--epochs` controls training horizon for each run; the quality-vs-time comparison is created by sampling each run's timeline at the same wall-clock budgets.
- `--time-grid` or `--time-grid-points` controls where quality is sampled on the time axis for the final comparison tables.

## QA Block Size Comparison

Use this script to compare different QA `subset_size` values with the same wall-clock sampling methodology.
It generates the same artifact structure as the quality-vs-wall-clock benchmark.

Example run:

```bash
python scripts/run_block_size_comparison.py \
  --model mlp \
  --dataset digits \
  --qa-samplers simulated,dwave \
  --subset-sizes 6,12,24,36 \
  --seeds 7,21,42 \
  --epochs 30 \
  --quality-metric test_metric \
  --quality-direction auto \
  --time-grid-points 20 \
  --output-dir results
```

## Overhead Communication Experiment

Use this script to profile QA step-time components and identify bottlenecks before publication.
It breaks the QA step into:

- `build_bqm_time_sec`
- `transfer_time_sec`
- `sampling_time_sec`
- `update_time_sec`

Then it computes percentage shares and plots `time share [%]` versus problem size (`subset_size`).

Example run:

```bash
python scripts/run_overhead_breakdown.py \
  --model mlp \
  --dataset digits \
  --qa-samplers simulated,dwave \
  --subset-sizes 6,12,24,36 \
  --seeds 7,21,42 \
  --epochs 20 \
  --output-dir results
```

Produced artifacts:

- `*_runs.json`, `*_runs.csv`: per-run absolute times and shares
- `*_summary.json`, `*_summary.csv`: aggregated means/std over seeds
- `*_<sampler>_share_vs_subset_size.png`: stacked share plot for bottleneck analysis

## Sampler Transition Analysis (simulated -> hybrid -> QPU)

Use this script to run the same QA hyperparameter grid across sampler backends and test distribution-level effects across many seeds.

It reports:

- run-level results for every `(config, sampler, seed)`
- per-sampler distribution statistics (`mean`, `std`, `var`, quantiles)
- bad-run analysis per shared config (to test whether QPU reduces the fraction of poor outcomes)
- plots: quality boxplot per sampler and bad-run-rate bar chart

Example run:

```bash
python scripts/run_sampler_transition_analysis.py \
  --model mlp \
  --dataset digits \
  --samplers simulated,hybrid,dwave \
  --subset-sizes 8,12 \
  --step-sizes 0.03,0.05 \
  --num-reads 100,200 \
  --seeds 7,21,42,84,168 \
  --epochs 20 \
  --quality-metric final_test_metric \
  --quality-direction auto \
  --output-dir results
```

## Where useful experiment artifacts are stored

- `mlruns/`: MLflow run directories (if experiments used MLflow)
- `data/`: input datasets (e.g., MNIST raw files)
- `notebooks/`: reproducible example notebooks

## Development notes

- Python compatibility: developed against Python 3.10+. Use a virtual environment.
- Smoke tests are available in `tests/test_smoke.py`.
- Run smoke tests with `python -m unittest tests.test_smoke`.
- If you modify the optimizer state logic, ensure acceptance/rejection semantics are
  preserved: proposed moment/state updates should normally only be committed after an
  accepted candidate.
