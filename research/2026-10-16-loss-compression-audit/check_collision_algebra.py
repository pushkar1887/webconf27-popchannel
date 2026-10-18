"""Synthetic algebra checks for a paper review; no recommender training or real-data result."""
import json
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
torch.set_default_dtype(torch.float64)


def uniformity(v):
    return (-2 * torch.pdist(F.normalize(v, dim=1)).square()).exp().mean().log()


def main():
    # Actual sharing: four distinct IDs share two antipodal buckets, two IDs per bucket.
    v = torch.tensor([[1., 0.], [-1., 0.]])
    g = torch.tensor([0, 0, 1, 1])
    actual = float(uniformity(v[g]))
    pi_with_replacement = .5
    rho = 1 / 3
    exact = math.log(rho + (1 - rho) * math.exp(-8))
    assert abs(actual - exact) < 1e-12
    assert actual < math.log(pi_with_replacement)

    # Derivative factorization for a fixed hard assignment and fixed batch.
    parameters = torch.tensor([[1., .3, -.2], [.1, 1., .4], [-.7, .2, 1.]], requires_grad=True)
    buckets = torch.tensor([0, 0, 0, 1, 2, 2])
    pair = torch.triu_indices(len(buckets), len(buckets), offset=1)
    x = F.normalize(parameters[buckets], dim=1)
    kernel = (-2 * (x[pair[0]] - x[pair[1]]).square().sum(dim=1)).exp()
    collided = buckets[pair[0]] == buckets[pair[1]]
    r = float(collided.double().mean())
    z = kernel[~collided].mean()
    total = kernel.mean().log()
    conditional = z.log()
    alpha = (1 - r) * z / (r + (1 - r) * z)
    grad_total = torch.autograd.grad(total, parameters, retain_graph=True)[0]
    grad_conditional = torch.autograd.grad(conditional, parameters, retain_graph=True)[0]
    error = float((grad_total - alpha.detach() * grad_conditional).abs().max())
    assert error < 1e-12
    rescaled_original = total / alpha.detach()
    grad_rescaled = torch.autograd.grad(rescaled_original, parameters, retain_graph=True)[0]
    rescaling_error = float((grad_rescaled - grad_conditional).abs().max())
    assert rescaling_error < 1e-12
    unique_loss = uniformity(parameters)
    grad_unique = torch.autograd.grad(unique_loss, parameters)[0]
    cosine = float(F.cosine_similarity(grad_unique.reshape(1, -1), grad_conditional.reshape(1, -1)))
    assert cosine < .9999  # Unweighted bucket deduplication changes the sampling measure.

    # A positive and a sampled negative sharing an item vector also affect BPR.
    user = torch.tensor([.2, -.4], requires_grad=True)
    item = torch.tensor([.7, .1], requires_grad=True)
    bpr = F.softplus(-((user * item).sum() - (user * item).sum()))
    gu, gi = torch.autograd.grad(bpr, [user, item])
    assert abs(float(bpr.detach()) - math.log(2)) < 1e-12
    assert float(gu.abs().max()) == float(gi.abs().max()) == 0

    # Same separate user/item collision masses; different mixed-volume penalty.
    a = np.array([.8, .2]); c = np.array([.8, .2])
    pi_sum = float(a @ a + c @ c)
    mixed_aligned, mixed_reversed = float((a + c) @ (a + c)), float((a + c[::-1]) @ (a + c[::-1]))
    assert abs(pi_sum - float(a @ a + c[::-1] @ c[::-1])) < 1e-12
    assert mixed_aligned != mixed_reversed

    # Normalized training losses are invariant to positive per-vector scaling;
    # raw-dot-product rankings need not be. Both inspected author repos use raw dot inference.
    q = torch.tensor([[.8, .6], [.7, math.sqrt(.51)]])
    scales = torch.tensor([[1.], [2.]])
    assert torch.allclose(F.normalize(q, dim=1), F.normalize(q * scales, dim=1))
    u = torch.tensor([1., 0.])
    assert int(torch.argmax(q @ u)) != int(torch.argmax((q * scales) @ u))

    # Initial bucket sharing need not survive neighbor-dependent LightGCN propagation.
    p = torch.tensor([.2, .3]); q0 = torch.tensor([1., 0.]); q1 = torch.tensor([0., 1.])
    out0, out1 = (p + q0) / 2, (p + q1) / 2
    assert not torch.allclose(out0, out1)

    result = {
        'status': 'synthetic_algebra_checks_passed',
        'scope': 'Deterministic mathematical checks only; no real data, loss-recovery experiment, or paper reproduction.',
        'finite_pair_floor_counterexample': {'batch_counts': [2, 2], 'pi': pi_with_replacement, 'rho': rho, 'actual_uniformity': actual, 'claimed_log_pi': math.log(pi_with_replacement), 'correct_spherical_floor': exact},
        'gradient_factorization': {'batch_rho': r, 'cross_bucket_kernel_mean': float(z.detach()), 'alpha': float(alpha.detach()), 'max_identity_error': error, 'adaptive_rescaling_max_gradient_error': rescaling_error, 'deduplicated_vs_multiplicity_preserved_gradient_cosine': cosine},
        'BPR_item_collision': {'loss': float(bpr.detach()), 'ranking_loss_gradients_zero': True},
        'mixed_volume_non_equivalence': {'separate_collision_mass_sum': pi_sum, 'mixed_volume_penalties': [mixed_aligned, mixed_reversed]},
        'raw_dot_score_scale_sensitivity_verified': True,
        'LightGCN_can_break_initial_alias_equality_verified': True,
    }
    (ROOT / 'algebra_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
