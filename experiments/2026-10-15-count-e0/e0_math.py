"""Numerical helpers for a throwaway research spike; no production method."""
import numpy as np


def bernoulli_kl(q, m):
    q = np.clip(np.asarray(q, dtype='float64'), 1e-12, 1 - 1e-12)
    m = np.clip(np.asarray(m, dtype='float64'), 1e-12, 1 - 1e-12)
    return q * np.log(q / m) + (1 - q) * np.log((1 - q) / (1 - m))


def tilted_cost(c, p, temperature):
    c, p = np.asarray(c, dtype='float64'), np.asarray(p, dtype='float64')
    if temperature is None:
        return c @ p
    if temperature <= 0:
        raise ValueError('Temperature must be positive')
    anchor = c.max(axis=-1, keepdims=True)
    return anchor[..., 0] + temperature * np.log(np.sum(p * np.exp((c - anchor) / temperature), axis=-1))


def context_mix(p, gamma, target=None):
    p = np.asarray(p, dtype='float64')
    if np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError('Invalid training context distribution')
    if not 0 <= gamma <= 1:
        raise ValueError('Invalid sweep coordinate')
    if target is not None:
        q = (1 - gamma) * p
        q[target] += gamma
        return q
    q = np.zeros_like(p)
    q[p > 0] = p[p > 0] ** (1 - gamma)
    return q / q.sum()


def distortion(profiles, groups, budget, p, temperature):
    profiles = np.clip(np.asarray(profiles, dtype='float64'), 1e-12, 1 - 1e-12)
    n = np.bincount(groups, minlength=budget).astype('float64')
    if np.any(n == 0):
        raise ValueError('Every group must be nonempty')
    sums = np.stack([np.bincount(groups, weights=profiles[:, c], minlength=budget) for c in range(profiles.shape[1])], axis=1)
    centers = sums / n[:, None]
    e = profiles * np.log(profiles) + (1 - profiles) * np.log1p(-profiles)
    es = np.stack([np.bincount(groups, weights=e[:, c], minlength=budget) for c in range(profiles.shape[1])], axis=1)
    ce = centers * np.log(centers) + (1 - centers) * np.log1p(-centers)
    costs = np.maximum(es / n[:, None] - ce, 0)
    return float(n @ tilted_cost(costs, p, temperature) / n.sum()), costs, centers


def capped_assignment(costs, cap):
    costs = np.asarray(costs)
    n, budget = costs.shape
    if budget * cap < n or budget > n:
        raise ValueError('Infeasible group capacity')
    groups = np.argmin(costs, axis=1)
    sizes = np.bincount(groups, minlength=budget)
    displaced = []
    # Keep the users for whom moving away would be most costly.
    for g in np.flatnonzero(sizes > cap):
        ids = np.flatnonzero(groups == g)
        alternatives = costs[ids].copy()
        alternatives[:, g] = np.inf
        regret = alternatives.min(axis=1) - costs[ids, g]
        release = ids[np.argsort(regret, kind='stable')[:len(ids) - cap]]
        displaced.extend(release.tolist())
        groups[release] = -1
        sizes[g] = cap
    for u in displaced:
        available = sizes < cap
        g = int(np.argmin(np.where(available, costs[u], np.inf)))
        groups[u] = g
        sizes[g] += 1
    for g in np.flatnonzero(sizes == 0):
        eligible = sizes[groups] > 1
        extra = costs[:, g] - costs[np.arange(n), groups]
        u = int(np.argmin(np.where(eligible, extra, np.inf)))
        sizes[groups[u]] -= 1
        groups[u] = g
        sizes[g] += 1
    if np.any(sizes < 1) or np.any(sizes > cap):
        raise AssertionError('Invalid repaired assignment')
    return groups.astype('int32')


def fitted_group_table(n, s, groups, budget, prior, alpha):
    gn = np.stack([np.bincount(groups, weights=n[:, c], minlength=budget) for c in range(n.shape[1])], axis=1)
    gs = np.stack([np.bincount(groups, weights=s[:, c], minlength=budget) for c in range(s.shape[1])], axis=1)
    return (gs + alpha * prior) / (gn + alpha)


def reweighted_loss(by_context, q):
    by_context, q = np.asarray(by_context), np.asarray(q)
    active = q > 0
    if not np.isfinite(by_context[active]).all():
        return None
    return float(by_context[active] @ q[active])
