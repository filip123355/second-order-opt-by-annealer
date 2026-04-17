from __future__ import annotations

from typing import Any
from time import perf_counter

import dimod


class VeloxQSampler:
    """Adapter exposing a dimod-like sampler interface over veloxq_sdk."""

    def __init__(
        self,
        backend: str = "h100_1",
        solver: str = "veloxq",
        num_rep: int | None = None,
        num_steps: int | None = None,
        discrete_version: bool | None = None,
        dt: float | None = None,
        wait_timeout: float | None = None,
        refresh_on_completion: bool = True,
    ) -> None:
        self.backend = backend
        self.solver = solver
        self.wait_timeout = wait_timeout
        self.refresh_on_completion = refresh_on_completion
        self._sdk = self._import_sdk()
        self._solver_instance = self._build_solver(
            backend=backend,
            solver=solver,
            num_rep=num_rep,
            num_steps=num_steps,
            discrete_version=discrete_version,
            dt=dt,
        )

    def _import_sdk(self) -> Any:
        try:
            import veloxq_sdk
        except ImportError as exc:
            raise ImportError(
                "veloxq_sdk is required for mode='veloxq'. Install it and configure VELOX_TOKEN."
            ) from exc
        return veloxq_sdk

    def _build_solver(
        self,
        backend: str,
        solver: str,
        num_rep: int | None,
        num_steps: int | None,
        discrete_version: bool | None,
        dt: float | None,
    ) -> Any:
        backend_map = {
            "h100_1": "VeloxQH100_1",
            "h100_2": "VeloxQH100_2",
            "plgrid_gh200": "PLGridGH200",
        }
        if backend not in backend_map:
            raise ValueError(f"Unsupported veloxQ backend '{backend}'.")

        backend_cls_name = backend_map[backend]
        backend_cls = getattr(self._sdk, backend_cls_name, None)
        if backend_cls is None:
            raise ValueError(
                f"Backend class '{backend_cls_name}' is unavailable in the installed veloxq_sdk."
            )

        backend_obj = backend_cls()

        if solver == "veloxq":
            solver_cls = getattr(self._sdk, "VeloxQSolver", None)
        elif solver == "sbm":
            solver_cls = getattr(self._sdk, "SBMSolver", None)
        else:
            raise ValueError(f"Unsupported veloxQ solver '{solver}'.")

        if solver_cls is None:
            raise ValueError(f"Solver '{solver}' is unavailable in the installed veloxq_sdk.")

        solver_obj = solver_cls(backend=backend_obj)

        parameters = getattr(solver_obj, "parameters", None)
        if parameters is not None:
            if num_rep is not None and hasattr(parameters, "num_rep"):
                parameters.num_rep = int(num_rep)
            if num_steps is not None and hasattr(parameters, "num_steps"):
                parameters.num_steps = int(num_steps)
            if discrete_version is not None and hasattr(parameters, "discrete_version"):
                parameters.discrete_version = bool(discrete_version)
            if dt is not None and hasattr(parameters, "dt"):
                parameters.dt = float(dt)

        return solver_obj

    def sample(self, bqm: dimod.BinaryQuadraticModel, **kwargs: Any) -> dimod.SampleSet:
        parameters = getattr(self._solver_instance, "parameters", None)
        num_reads = kwargs.get("num_reads")
        if parameters is not None and num_reads is not None and hasattr(parameters, "num_rep"):
            parameters.num_rep = int(num_reads)

        num_steps = kwargs.get("num_steps")
        if parameters is not None and num_steps is not None and hasattr(parameters, "num_steps"):
            parameters.num_steps = int(num_steps)

        transfer_start = perf_counter()
        file_obj = self._sdk.File.from_bqm(bqm)
        job = self._solver_instance.submit(file_obj)
        transfer_time_sec = float(perf_counter() - transfer_start)

        backend_sampling_start = perf_counter()
        if self.wait_timeout is None:
            job.wait_for_completion(refresh=self.refresh_on_completion)
        else:
            job.wait_for_completion(timeout=self.wait_timeout, refresh=self.refresh_on_completion)
        backend_sampling_time_sec = float(perf_counter() - backend_sampling_start)

        result = job.result
        if isinstance(result.info, dict):
            result.info.setdefault("solver", self.solver)
            result.info.setdefault("backend", self.backend)
            result.info.setdefault("job_id", str(getattr(job, "id", "")))
            result.info.setdefault("transfer_time_sec", transfer_time_sec)
            result.info.setdefault("backend_sampling_time_sec", backend_sampling_time_sec)

        return result