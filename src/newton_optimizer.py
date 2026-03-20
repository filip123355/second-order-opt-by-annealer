import torch
from torch import Tensor
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from typing import Callable


class NewtonOptimizer(torch.optim.Optimizer):
    """Implements the pure Newton-Raphson optimization algorithm.

    The closure passed to ``step`` must return a scalar loss tensor and must
    not call ``loss.backward()``. Gradients are computed internally.

    This implementation builds the exact dense Hessian with autograd and solves
    the linear system ``H d = g`` directly, without trust-region or damping
    regularization. The ``damping`` and ``trust_region_radius`` arguments are
    kept only for backwards-compatible construction and are ignored.
    """

    def __init__(self, 
            params,
            lr: float | Tensor = 1.0,
            max_iter: int = 1,
            tolerance_grad: float = 1e-7,
            damping: float | Tensor | None = None,
    ) -> None:
        self.defaults = dict(
            lr=lr,
            max_iter=max_iter,
            tolerance_grad=tolerance_grad,
            damping=damping,
        )
        super(NewtonOptimizer, self).__init__(params, self.defaults)

    def _compute_exact_hessian(
        self,
        grad_vec: Tensor,
        params: list[Tensor],
    ) -> Tensor:
        hessian = torch.empty(
            (grad_vec.numel(), grad_vec.numel()),
            device=grad_vec.device,
            dtype=grad_vec.dtype,
        )

        for row_index in range(grad_vec.numel()):
            second_grads = torch.autograd.grad(
                grad_vec[row_index],
                params,
                retain_graph=row_index + 1 < grad_vec.numel(),
            )
            hessian[row_index] = parameters_to_vector(second_grads)

        return 0.5 * (hessian + hessian.T)

    def step(self, loss_or_closure: Tensor | Callable[[], Tensor]):
        params = [
            param
            for group in self.param_groups
            for param in group["params"]
            if param.requires_grad
        ]

        if not params:
            return {"loss": None}

        closure = loss_or_closure if callable(loss_or_closure) else None
        loss = loss_or_closure if not callable(loss_or_closure) else None

        for _ in range(self.defaults["max_iter"]):
            current_params = parameters_to_vector(params).detach().clone()

            if closure is not None:
                loss = closure()

            grads = torch.autograd.grad(loss, params, create_graph=True)
            grad_vec = parameters_to_vector(grads)

            if torch.linalg.vector_norm(grad_vec) < self.defaults["tolerance_grad"]:
                break

            hessian = self._compute_exact_hessian(grad_vec, params)
            damp_ = self.defaults["damping"] * torch.eye(hessian.shape[0], device=hessian.device)
            direction = torch.linalg.solve(hessian + damp_, grad_vec)

            with torch.no_grad():
                vector_to_parameters(
                    current_params - self.defaults["lr"] * direction,
                    params,
                )

        return {
            "loss": float(loss.item()),
        }
