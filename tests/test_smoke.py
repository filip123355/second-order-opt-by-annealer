import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn as nn
from dimod import ExactSolver

from src.quadratic_annealing_optimizer import QuadraticAnnealingOptimizer
from src.utils import data_load_and_prep

class SmokeTests(unittest.TestCase):
    def test_diabetes_split_without_stratify(self):
        captured = {}

        def fake_split(X, y, *args, **kwargs):
            captured.update(kwargs)
            return X, X, y, y

        with patch("src.utils.train_test_split", side_effect=fake_split):
            data_load_and_prep(dataset_name="diabetes", random_state=42, batch_size=16)

        self.assertIn("stratify", captured)
        self.assertIsNone(captured["stratify"])

    def test_optimizer_step_returns_backend_metadata(self):
        model = nn.Linear(4, 3)
        optimizer = QuadraticAnnealingOptimizer(
            sampler=ExactSolver(),
            model=model,
            subset_size=2,
            step_size=0.01,
            num_reads=4,
        )

        features = torch.randn(6, 4)
        targets = torch.randint(0, 3, (6,))
        step_info = optimizer.step(features, targets, nn.CrossEntropyLoss())

        self.assertIn("sampler_name", step_info)
        self.assertIn("qpu_access_time", step_info)

    def test_grid_script_writes_json_and_csv(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_experiment_grid.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                str(script),
                "--model",
                "mlp",
                "--dataset",
                "iris",
                "--optimizer",
                "adam",
                "--epochs",
                "1",
                "--batch-size",
                "full",
                "--seeds",
                "1",
                "--output-dir",
                tmpdir,
                "--output-prefix",
                "smoke",
                "--max-runs",
                "1",
                "--quiet",
            ]
            subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)

            output_files = list(Path(tmpdir).glob("smoke_*.json"))
            self.assertTrue(output_files)
            csv_files = list(Path(tmpdir).glob("smoke_*.csv"))
            self.assertTrue(csv_files)

    def test_optimizer_comparison_script_writes_json_and_csv(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_optimizer_comparison.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                str(script),
                "--model",
                "mlp",
                "--dataset",
                "iris",
                "--optimizers",
                "qa,adam",
                "--qa-samplers",
                "simulated",
                "--epochs",
                "1",
                "--batch-size",
                "full",
                "--seed",
                "1",
                "--output-dir",
                tmpdir,
                "--output-prefix",
                "compare_smoke",
                "--quiet",
            ]
            subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)

            output_files = list(Path(tmpdir).glob("compare_smoke_*.json"))
            self.assertTrue(output_files)
            csv_files = list(Path(tmpdir).glob("compare_smoke_*.csv"))
            self.assertTrue(csv_files)

    def test_quality_vs_wallclock_script_writes_expected_artifacts(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_quality_vs_wallclock.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                str(script),
                "--model",
                "mlp",
                "--dataset",
                "iris",
                "--optimizers",
                "adam,qa",
                "--qa-samplers",
                "simulated",
                "--epochs",
                "1",
                "--batch-size",
                "full",
                "--seeds",
                "1",
                "--time-grid",
                "0.0,1.0",
                "--output-dir",
                tmpdir,
                "--output-prefix",
                "qvw_smoke",
                "--quiet",
            ]
            subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)

            self.assertTrue(list(Path(tmpdir).glob("qvw_smoke_*_runs.json")))
            self.assertTrue(list(Path(tmpdir).glob("qvw_smoke_*_runs.csv")))
            self.assertTrue(list(Path(tmpdir).glob("qvw_smoke_*_timeline.json")))
            self.assertTrue(list(Path(tmpdir).glob("qvw_smoke_*_timeline.csv")))
            self.assertTrue(list(Path(tmpdir).glob("qvw_smoke_*_budget_raw.json")))
            self.assertTrue(list(Path(tmpdir).glob("qvw_smoke_*_budget_raw.csv")))
            self.assertTrue(list(Path(tmpdir).glob("qvw_smoke_*_budget_summary.json")))
            self.assertTrue(list(Path(tmpdir).glob("qvw_smoke_*_budget_summary.csv")))

    def test_block_size_comparison_script_writes_expected_artifacts(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_block_size_comparison.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                str(script),
                "--model",
                "mlp",
                "--dataset",
                "iris",
                "--qa-samplers",
                "simulated",
                "--subset-sizes",
                "2,4",
                "--epochs",
                "1",
                "--batch-size",
                "full",
                "--seeds",
                "1",
                "--time-grid",
                "0.0,1.0",
                "--output-dir",
                tmpdir,
                "--output-prefix",
                "block_smoke",
                "--quiet",
            ]
            subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)

            self.assertTrue(list(Path(tmpdir).glob("block_smoke_*_runs.json")))
            self.assertTrue(list(Path(tmpdir).glob("block_smoke_*_runs.csv")))
            self.assertTrue(list(Path(tmpdir).glob("block_smoke_*_timeline.json")))
            self.assertTrue(list(Path(tmpdir).glob("block_smoke_*_timeline.csv")))
            self.assertTrue(list(Path(tmpdir).glob("block_smoke_*_budget_raw.json")))
            self.assertTrue(list(Path(tmpdir).glob("block_smoke_*_budget_raw.csv")))
            self.assertTrue(list(Path(tmpdir).glob("block_smoke_*_budget_summary.json")))
            self.assertTrue(list(Path(tmpdir).glob("block_smoke_*_budget_summary.csv")))

    def test_overhead_breakdown_script_writes_expected_artifacts(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_overhead_breakdown.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                str(script),
                "--model",
                "mlp",
                "--dataset",
                "iris",
                "--qa-samplers",
                "simulated",
                "--subset-sizes",
                "2,4",
                "--epochs",
                "1",
                "--batch-size",
                "full",
                "--seeds",
                "1",
                "--output-dir",
                tmpdir,
                "--output-prefix",
                "overhead_smoke",
                "--quiet",
            ]
            subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)

            self.assertTrue(list(Path(tmpdir).glob("overhead_smoke_*_runs.json")))
            self.assertTrue(list(Path(tmpdir).glob("overhead_smoke_*_runs.csv")))
            self.assertTrue(list(Path(tmpdir).glob("overhead_smoke_*_summary.json")))
            self.assertTrue(list(Path(tmpdir).glob("overhead_smoke_*_summary.csv")))
            self.assertTrue(list(Path(tmpdir).glob("overhead_smoke_*_share_vs_subset_size.png")))

    def test_sampler_transition_analysis_script_writes_expected_artifacts(self):
        repo_root = Path(__file__).resolve().parents[1]
        script = repo_root / "scripts" / "run_sampler_transition_analysis.py"

        with tempfile.TemporaryDirectory() as tmpdir:
            command = [
                sys.executable,
                str(script),
                "--model",
                "mlp",
                "--dataset",
                "iris",
                "--samplers",
                "simulated,hybrid",
                "--subset-sizes",
                "2",
                "--step-sizes",
                "0.05",
                "--num-reads",
                "10",
                "--epochs",
                "1",
                "--batch-size",
                "full",
                "--seeds",
                "1,2",
                "--output-dir",
                tmpdir,
                "--output-prefix",
                "transition_smoke",
                "--quiet",
            ]
            subprocess.run(command, cwd=repo_root, check=True, capture_output=True, text=True)

            self.assertTrue(list(Path(tmpdir).glob("transition_smoke_*_runs.json")))
            self.assertTrue(list(Path(tmpdir).glob("transition_smoke_*_runs.csv")))
            self.assertTrue(list(Path(tmpdir).glob("transition_smoke_*_sampler_summary.json")))
            self.assertTrue(list(Path(tmpdir).glob("transition_smoke_*_sampler_summary.csv")))
            self.assertTrue(list(Path(tmpdir).glob("transition_smoke_*_bad_run_analysis.json")))
            self.assertTrue(list(Path(tmpdir).glob("transition_smoke_*_bad_run_analysis.csv")))
            self.assertTrue(list(Path(tmpdir).glob("transition_smoke_*_quality_distribution_boxplot.png")))
            self.assertTrue(list(Path(tmpdir).glob("transition_smoke_*_bad_run_rate.png")))


if __name__ == "__main__":
    unittest.main()
