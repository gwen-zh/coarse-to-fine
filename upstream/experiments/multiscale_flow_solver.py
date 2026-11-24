"""ODE solver controls for the frozen multiscale generator; no noise scaling."""
import torch

from hierarchical_ensemble_flow import component_seed


@torch.no_grad()
def integrate_velocity(initial, velocity, nfe, method="euler"):
    if method not in {"euler", "heun"} or nfe < 1:
        raise ValueError("Expected a supported solver and positive NFE")
    if method == "heun" and nfe % 2:
        raise ValueError("Heun requires an even velocity-evaluation budget")
    intervals = nfe if method == "euler" else nfe // 2
    value = initial.clone()
    for index in range(intervals):
        t = index / intervals
        first = velocity(value, t)
        if method == "euler":
            value = value + first / intervals
        else:
            proposal = value + first / intervals
            second = velocity(proposal, (index + 1) / intervals)
            value = value + (first + second) / (2 * intervals)
        if not torch.isfinite(value).all():
            raise RuntimeError("Nonfinite ODE state; do not drop or retry a sample")
    return value


@torch.no_grad()
def sample_with_coarse_solver(model, codec, aa, seed, coarse_nfe=16,
                             method="euler", fine_nfe=16):
    generator = torch.Generator(device=aa.device).manual_seed(component_seed(seed, "coarse"))
    initial = torch.randn(((len(aa) + 3) // 4, model.coarse_dim),
                          device=aa.device, generator=generator)
    normalized = integrate_velocity(initial,
        lambda state, time: model.coarse_velocity(aa, state, time), coarse_nfe, method)
    coarse = codec.project_coarse(model.denormalize_coarse(normalized), len(aa))
    fine = model.sample_fine(aa, coarse, component_seed(seed, "fine"), fine_nfe)
    return coarse, fine
