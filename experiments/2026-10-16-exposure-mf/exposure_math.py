"""Numerical helpers for a throwaway exposure-evaluation pilot."""
import math
import numpy as np


def loss(y, score):
    score = np.asarray(score, dtype='float64')
    return np.logaddexp(0, score) - np.asarray(y) * score


def user_means(values, uid, nusers):
    n = np.bincount(uid, minlength=nusers)
    sm = np.bincount(uid, weights=values, minlength=nusers)
    return np.divide(sm, n, out=np.full(nusers, np.nan), where=n > 0)


def user_auc(y, score, uid, nusers):
    order = np.lexsort((score, uid))
    u, z, yy = uid[order], np.asarray(score)[order], y[order]
    user_starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
    user_sizes = np.diff(np.r_[user_starts, len(u)])
    rank = np.arange(len(u)) - np.repeat(user_starts, user_sizes) + 1.
    tie_starts = np.flatnonzero(np.r_[True, (u[1:] != u[:-1]) | (z[1:] != z[:-1])])
    tie_ends = np.r_[tie_starts[1:], len(u)]
    average_rank = .5 * (rank[tie_starts] + rank[tie_ends - 1])
    rank = np.repeat(average_rank, tie_ends - tie_starts)
    positives = np.bincount(u, weights=yy, minlength=nusers)
    negatives = np.bincount(u, minlength=nusers) - positives
    ranks_positive = np.bincount(u, weights=rank * yy, minlength=nusers)
    numerator = ranks_positive - positives * (positives + 1) / 2
    denominator = positives * negatives
    return np.divide(numerator, denominator, out=np.full(nusers, np.nan), where=denominator > 0)


def fit_intercept(y, score, uid, nusers, user_weights=None):
    n = np.bincount(uid, minlength=nusers)
    if np.any(n == 0):
        raise ValueError('Calibration roster contains a missing user')
    weights = np.ones(nusers) if user_weights is None else np.asarray(user_weights, dtype='float64')
    ew = weights[uid] / n[uid]
    score = np.asarray(score, dtype='float64')
    lo, hi = -40., 40.
    for _ in range(70):
        mid = (lo + hi) / 2
        pr = 1 / (1 + np.exp(-np.clip(score + mid, -80, 80)))
        derivative = float(ew @ (pr - y))
        if derivative > 0:
            hi = mid
        else:
            lo = mid
    shift = (lo + hi) / 2
    pr = 1 / (1 + np.exp(-np.clip(score + shift, -80, 80)))
    g = user_means(pr - y, uid, nusers)
    h = float(np.average(user_means(pr * (1 - pr), uid, nusers), weights=weights))
    if not h > 1e-10:
        raise ValueError('Degenerate calibration curvature')
    return shift, g, h


def calibration_if(weights, g, h):
    return -(weights @ g - g.sum()) / (len(g) * h)


def matched_capacity(nusers, dimension, rank):
    dense = nusers * rank + rank * dimension
    budget = math.ceil(dense / dimension)
    return budget, dense, budget * dimension
