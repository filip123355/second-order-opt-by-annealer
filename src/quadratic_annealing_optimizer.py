import torch.nn as nn
import torch
import dimod
from time import perf_counter

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from dimod import BinaryQuadraticModel, ExactSolver
from .losses import RidgeLoss, SVMSquaredHingeLoss
class QuadraticAnnealingOptimizer:
    """An optimizer that uses a quadratic approximation of the loss landscape to construct a binary quadratic model, 
    which is then optimized using a quantum annealer or a classical sampler. The optimizer selects a subset of parameters 
    based on the magnitude of their gradients or randomly, and constructs a BQM that approximates the loss landscape in 
    the neighborhood of the current parameters. The BQM is then optimized using the provided sampler, and the candidate 
    parameters are evaluated on the loss function. If the candidate parameters yield a lower loss, they are accepted; 
    otherwise, the optimizer reverts to the original parameters.
    """

    def __init__(
        self,
        sampler,
        model: nn.Module,
        device: str = "cpu",
        subset_size: int = 12,
        step_size: float = 0.05,
        num_reads: int = 100,
        selection: str | float = "topk",
        beta: float | None = None,
    ):
        self.sampler = sampler
        self.model = model
        self.device = torch.device(device)
        self.subset_size = subset_size
        self.step_size = step_size
        self.num_reads = num_reads
        self.selection = selection
        self.defaults = {
            "device": device,
            "subset_size": subset_size,
            "step_size": step_size,
            "num_reads": num_reads,
            "selection": selection,
            "beta": beta,
        }
        self.param_groups = [self.defaults]
        self.momentum: torch.Tensor | None = None

    def _extract_backend_metadata(self, response: dimod.SampleSet) -> dict[str, float | str | None]:
        info = response.info if isinstance(response.info, dict) else {}
        timing = info.get("timing", {}) if isinstance(info.get("timing", {}), dict) else {}

        def _first_float(*keys: str) -> float | None:
            for key in keys:
                value = None
                if key in info:
                    value = info.get(key)
                elif key in timing:
                    value = timing.get(key)
                if value is None:
                    continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return None

        return {
            "sampler_name": type(self.sampler).__name__,
            "solver_id": str(info.get("problem_id") or info.get("solver") or info.get("solver_id") or ""),
            "qpu_access_time": _first_float("qpu_access_time", "qpu_access_time_us"),
            "qpu_sampling_time": _first_float("qpu_sampling_time", "qpu_sampling_time_us"),
            "qpu_readout_time": _first_float("qpu_readout_time", "qpu_readout_time_us"),
            "qpu_anneal_time_per_sample": _first_float(
                "qpu_anneal_time_per_sample",
                "qpu_anneal_time_per_sample_us",
            ),
            "qpu_delay_time_per_sample": _first_float(
                "qpu_delay_time_per_sample",
                "qpu_delay_time_per_sample_us",
            ),
        }

    def _extract_sampling_breakdown(
        self,
        response: dimod.SampleSet,
        sample_total_time_sec: float,
    ) -> tuple[float, float]:
        """Return (transfer_time_sec, sampling_time_sec) for one sampler call."""
        info = response.info if isinstance(response.info, dict) else {}

        def _as_float(value: object) -> float | None:
            try:
                if value is None:
                    return None
                return float(value)
            except (TypeError, ValueError):
                return None

        transfer_time = _as_float(info.get("transfer_time_sec"))
        backend_sampling_time = _as_float(info.get("backend_sampling_time_sec"))
        if transfer_time is not None and backend_sampling_time is not None:
            return max(0.0, transfer_time), max(0.0, backend_sampling_time)

        # D-Wave exposes QPU timing mostly in microseconds. Convert if available and
        # estimate communication/overhead as sample_total - qpu_access.
        qpu_access = _as_float(info.get("qpu_access_time"))
        if qpu_access is None and isinstance(info.get("timing"), dict):
            qpu_access = _as_float(info["timing"].get("qpu_access_time"))

        qpu_access_us = _as_float(info.get("qpu_access_time_us"))
        if qpu_access_us is None and isinstance(info.get("timing"), dict):
            qpu_access_us = _as_float(info["timing"].get("qpu_access_time_us"))

        if qpu_access is not None:
            sampling_time_sec = max(0.0, qpu_access * 1e-6)
            transfer_time_sec = max(0.0, sample_total_time_sec - sampling_time_sec)
            return transfer_time_sec, sampling_time_sec

        if qpu_access_us is not None:
            sampling_time_sec = max(0.0, qpu_access_us * 1e-6)
            transfer_time_sec = max(0.0, sample_total_time_sec - sampling_time_sec)
            return transfer_time_sec, sampling_time_sec

        # Fallback for local/classical samplers where transfer is effectively zero.
        return 0.0, max(0.0, sample_total_time_sec)

    def _selected_indices(self, grad_vec: torch.Tensor) -> torch.Tensor:

        """
        Function to select the subset of indices that will be included in an optimization step.
        """
        block_size = min(self.subset_size, grad_vec.numel())
        if isinstance(self.selection, str):
            if self.selection == "topk":
                return torch.topk(grad_vec.abs(), k=block_size).indices
            if self.selection == "random":
                return torch.randperm(grad_vec.numel(), device=grad_vec.device)[:block_size]
        elif isinstance(self.selection, float):
            if 0 < self.selection < 1:
                num_elements = grad_vec.numel()
                top_k = int(min(self.selection * block_size, num_elements))

                top_k_inds = torch.topk(grad_vec.abs(), k=top_k).indices

                random_k = block_size - top_k
                if random_k > 0:
                    mask = torch.ones(num_elements, dtype=torch.bool, device=grad_vec.device)
                    mask[top_k_inds] = False  
                    
                    remaining_inds = torch.where(mask)[0]
                    
                    perm = torch.randperm(remaining_inds.size(0), device=grad_vec.device)
                    random_inds = remaining_inds[perm[:random_k]]
                    
                    return torch.cat([top_k_inds, random_inds])
                
                return top_k_inds
            else:
                raise ValueError("selection must be a float in (0, 1)")

    def quadratic_model(self, 
                        loss: torch.Tensor,
    ) -> list[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Function to compute the quadratic approximation of the loss landscape in the neighborhood of the current parameters.
        """

        params = [param for param in self.model.parameters() if param.requires_grad]
        grads = torch.autograd.grad(loss, params, create_graph=True)
        grad_vec = parameters_to_vector(grads)
        selected_indices = self._selected_indices(grad_vec)
        grad_block = grad_vec[selected_indices]

        hessian_block = torch.zeros(
            (selected_indices.numel(), selected_indices.numel()),
            device=self.device,
        )
        
        for row_index, param_index in enumerate(selected_indices):
            second_grads = torch.autograd.grad(
                grad_vec[param_index],
                params,
                retain_graph=row_index + 1 < selected_indices.numel(),
            )
            second_vec = parameters_to_vector(second_grads)
            hessian_block[row_index] = second_vec[selected_indices]

        hessian_block = 0.5 * (hessian_block + hessian_block.T) # Hermitization

        return (grad_vec.detach(), 
                selected_indices.detach(), 
                grad_block.detach(), 
                hessian_block.detach(),
                )

    def build_bqm(
        self,
        selected_indices: torch.Tensor,
        grad_block: torch.Tensor,
        hessian_block: torch.Tensor
    ) -> BinaryQuadraticModel:
        """ 
        Function to build a binary quadratic model from the selected gradient and Hessian blocks. 
        """
        step_sq = self.step_size ** 2
        linear = {}
        quadratic = {}
        offset = 0.0

        for local_index, global_index in enumerate(selected_indices.tolist()):
            gradient_value = float(grad_block[local_index].item())
            curvature_value = float(hessian_block[local_index, local_index].item())
            linear[int(global_index)] = 2.0 * self.step_size * gradient_value
            offset += -self.step_size * gradient_value
            offset += 0.5 * step_sq * curvature_value

        for left in range(selected_indices.numel()):
            for right in range(left + 1, selected_indices.numel()):
                coupling = float(hessian_block[left, right].item())
                if abs(coupling) < 1e-12:
                    continue

                left_index = int(selected_indices[left].item())
                right_index = int(selected_indices[right].item())
                linear[left_index] = linear.get(left_index, 0.0) - 2.0 * step_sq * coupling
                linear[right_index] = linear.get(right_index, 0.0) - 2.0 * step_sq * coupling
                quadratic[(left_index, right_index)] = 4.0 * step_sq * coupling
                offset += step_sq * coupling

        return BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)

    def sample_bqm(self, 
                   bqm: BinaryQuadraticModel,
    ) -> dimod.SampleSet:
        """ 
        Sampling from BQM according to chosen sampler.
        """
        if isinstance(self.sampler, ExactSolver):
            return self.sampler.sample(bqm)
        return self.sampler.sample(bqm, num_reads=self.num_reads)

    def step(self, 
             features: torch.Tensor, 
             targets: torch.Tensor, 
             loss_fn: nn.Module,
    ) -> dict[str, float | bool | int | str | None]:
        """ 
        Single optimization step.
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(features)
        loss = loss_fn(logits, targets, self.model) if (
            isinstance(loss_fn, RidgeLoss) or isinstance(
                loss_fn, SVMSquaredHingeLoss)) else loss_fn(logits, targets)

        params = [param for param in self.model.parameters() if param.requires_grad]
        current_params = parameters_to_vector(params).detach().clone()
        if self.momentum is None or self.momentum.shape != current_params.shape or self.momentum.device != current_params.device or self.momentum.dtype != current_params.dtype:
            self.momentum = torch.zeros_like(current_params)

        model_start = perf_counter()
        _, selected_indices, grad_block, hessian_block = self.quadratic_model(loss)
        quadratic_model_time_sec = float(perf_counter() - model_start)

        build_bqm_start = perf_counter()
        bqm = self.build_bqm(selected_indices, grad_block, hessian_block)
        build_bqm_time_sec = float(perf_counter() - build_bqm_start)

        sample_start = perf_counter()
        response = self.sample_bqm(bqm)
        sample_total_time_sec = float(perf_counter() - sample_start)
        backend_metadata = self._extract_backend_metadata(response)
        transfer_time_sec, sampling_time_sec = self._extract_sampling_breakdown(
            response=response,
            sample_total_time_sec=sample_total_time_sec,
        )

        update_start = perf_counter()
        delta = torch.zeros_like(current_params)
        for parameter_index in selected_indices.tolist():
            bit_value = response.first.sample[int(parameter_index)]
            delta_value = current_params.new_tensor(self.step_size if bit_value else -self.step_size)
            beta = self.defaults.get("beta", None)
            if beta is not None:
                delta_value = delta_value + float(beta) * self.momentum[int(parameter_index)]
            delta[int(parameter_index)] = delta_value

        with torch.no_grad():
            candidate_params = current_params + delta.to(current_params.device)
            vector_to_parameters(candidate_params, params)

            candidate_loss = (float(loss_fn(self.model(features), targets, self.model).item()) 
                              if (isinstance(loss_fn, RidgeLoss) or isinstance(loss_fn, SVMSquaredHingeLoss))
                              else float(loss_fn(self.model(features), targets).item()))

            if candidate_loss > float(loss.item()):
                vector_to_parameters(current_params, params)
                accepted = False
                effective_loss = float(loss.item())
            else:
                accepted = True
                effective_loss = candidate_loss

            if accepted and self.defaults.get("beta", None) is not None:
                self.momentum[selected_indices] = delta[selected_indices]

        update_time_sec = float(perf_counter() - update_start)

        result: dict[str, float | bool | int | str | None] = {
            "loss": effective_loss,
            "quadratic_energy": float(response.first.energy),
            "accepted": accepted,
            "selected_variables": int(selected_indices.numel()),
            "quadratic_model_time_sec": quadratic_model_time_sec,
            "build_bqm_time_sec": build_bqm_time_sec,
            "transfer_time_sec": transfer_time_sec,
            "sampling_time_sec": sampling_time_sec,
            "sample_total_time_sec": sample_total_time_sec,
            "update_time_sec": update_time_sec,
        }
        result.update(backend_metadata)
        return result
    

if __name__ == "__main__":
    QuadraticAnnealingOptimizer(
        sampler=ExactSolver(),
        model=nn.Linear(10, 1),
    )
    