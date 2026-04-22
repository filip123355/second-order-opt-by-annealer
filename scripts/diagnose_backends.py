#!/usr/bin/env python3
"""Lightweight diagnostics for available annealing backends.

The script intentionally uses a tiny 2-variable BQM and minimal sampling
configuration to reduce backend usage/cost.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import dimod

from src.utils import build_sampler


DEFAULT_MODES = [
    "exact",
    "simulated",
    "gpu_simulated",
    "dwave",
    "veloxq",
]


def _tiny_bqm() -> dimod.BinaryQuadraticModel:
    # Small non-trivial problem (2 vars + coupling).
    linear = {0: -0.1, 1: 0.2}
    quadratic = {(0, 1): -0.3}
    return dimod.BinaryQuadraticModel(linear, quadratic, 0.0, dimod.BINARY)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_basic_metrics(sampleset: dimod.SampleSet) -> dict[str, Any]:
    info = sampleset.info if isinstance(sampleset.info, dict) else {}
    timing = info.get("timing", {}) if isinstance(info.get("timing"), dict) else {}
    metrics: dict[str, Any] = {
        "energy": _safe_float(sampleset.first.energy),
        "num_variables": int(len(sampleset.variables)),
        "num_samples": int(len(sampleset.record)),
        "solver_id": str(info.get("problem_id") or info.get("solver") or info.get("solver_id") or ""),
        "qpu_access_time": _safe_float(info.get("qpu_access_time") or timing.get("qpu_access_time")),
        "qpu_sampling_time": _safe_float(info.get("qpu_sampling_time") or timing.get("qpu_sampling_time")),
        "qpu_readout_time": _safe_float(info.get("qpu_readout_time") or timing.get("qpu_readout_time")),
    }
    return metrics


def diagnose_mode(mode: str, num_reads: int, timeout_sec: float | None) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "mode": mode,
        "status": "ok",
        "sampler": "",
        "fallback": False,
        "error": None,
        "elapsed_sec": None,
        "metrics": {},
    }

    try:
        sampler = build_sampler(mode=mode)
        sampler_name = type(sampler).__name__
        result["sampler"] = sampler_name

        # If the SDK supports timeout (VeloxQSampler), use it to avoid long waits.
        if timeout_sec is not None and hasattr(sampler, "wait_timeout"):
            setattr(sampler, "wait_timeout", float(timeout_sec))

        # build_sampler("dwave") and build_sampler("hybrid") currently fallback to
        # simulated annealing on failure; detect and report it explicitly.
        if mode in {"dwave"} and sampler_name == "SimulatedAnnealingSampler":
            result["status"] = "fallback"
            result["fallback"] = True
            result["error"] = f"{mode} unavailable, fell back to SimulatedAnnealingSampler"
            result["elapsed_sec"] = float(time.perf_counter() - started)
            return result

        bqm = _tiny_bqm()
        if mode == "veloxq":
            sampler.parameters.num_rep = num_reads
            sampleset = sampler.sample(bqm)
        elif mode == "exact":
            sampleset = sampler.sample(bqm)

        result["metrics"] = _extract_basic_metrics(sampleset)
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
    finally:
        result["elapsed_sec"] = float(time.perf_counter() - started)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lightweight diagnostics for available annealing backends."
    )
    parser.add_argument(
        "--modes",
        default=",".join(DEFAULT_MODES),
        help="Comma-separated backend modes to check.",
    )
    parser.add_argument("--num-reads", type=int, default=1, help="Sampling reads for non-hybrid modes.")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=20.0,
        help="Optional timeout for backends exposing wait_timeout (e.g., VeloxQSampler).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only.",
    )
    args = parser.parse_args()

    modes = [mode.strip() for mode in str(args.modes).split(",") if mode.strip()]
    results = [diagnose_mode(mode, num_reads=args.num_reads, timeout_sec=args.timeout_sec) for mode in modes]

    summary = {
        "ok": sum(1 for row in results if row["status"] == "ok"),
        "fallback": sum(1 for row in results if row["status"] == "fallback"),
        "error": sum(1 for row in results if row["status"] == "error"),
        "total": len(results),
    }
    payload = {"summary": summary, "results": results}

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print("Backend diagnostics summary:")
    print(json.dumps(summary, indent=2))
    print()
    for row in results:
        print(f"- mode={row['mode']} status={row['status']} sampler={row['sampler']} elapsed={row['elapsed_sec']:.3f}s")
        if row["error"]:
            print(f"  error: {row['error']}")
        if row["metrics"]:
            print(f"  metrics: {json.dumps(row['metrics'])}")


if __name__ == "__main__":
    main()
