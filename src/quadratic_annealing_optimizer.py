import torch.nn as nn
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from dimod import BinaryQuadraticModel, ExactSolver
import dimod



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
        selection: str = "topk",
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
        }
        self.param_groups = [self.defaults]

    def _selected_indices(self, grad_vec: torch.Tensor) -> torch.Tensor:

        """
        Function to select the subset of indices that will be included in an optimization step.
        """
        block_size = min(self.subset_size, grad_vec.numel())
        if self.selection == "topk":
            return torch.topk(grad_vec.abs(), k=block_size).indices
        if self.selection == "random":
            return torch.randperm(grad_vec.numel(), device=grad_vec.device)[:block_size]
        raise ValueError("selection must be either 'topk' or 'random'")

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
        
        # TODO: Hessian computation has to be optimized for GPU in future. Now it does not matter for small models
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
        hessian_block: torch.Tensor,
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
    ) -> dict[str, float | bool | int]:
        """ 
        Single optimization step.
        """
        self.model.zero_grad(set_to_none=True)
        logits = self.model(features)
        loss = loss_fn(logits, targets)

        current_params = parameters_to_vector(self.model.parameters()).detach().clone()
        _, selected_indices, grad_block, hessian_block = self.quadratic_model(loss)
        bqm = self.build_bqm(selected_indices, grad_block, hessian_block)
        response = self.sample_bqm(bqm)

        delta = torch.zeros_like(current_params)
        for parameter_index in selected_indices.tolist():
            bit_value = response.first.sample[int(parameter_index)]
            delta[int(parameter_index)] = self.step_size if bit_value else -self.step_size

        with torch.no_grad():
            candidate_params = current_params + delta.to(current_params.device)
            vector_to_parameters(candidate_params, self.model.parameters())
            candidate_loss = float(loss_fn(self.model(features), targets).item())

            if candidate_loss > float(loss.item()):
                vector_to_parameters(current_params, self.model.parameters())
                accepted = False
                effective_loss = float(loss.item())
            else:
                accepted = True
                effective_loss = candidate_loss

        return {
            "loss": effective_loss,
            "quadratic_energy": float(response.first.energy),
            "accepted": accepted,
            "selected_variables": int(selected_indices.numel()),
        }
    
    def zero_grad(self):
        pass

if __name__ == "__main__":
    QuadraticAnnealingOptimizer(
        sampler=ExactSolver(),
        model=nn.Linear(10, 1),
    )
    