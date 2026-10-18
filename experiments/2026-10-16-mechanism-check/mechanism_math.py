"""Small numerical contracts for a throwaway research diagnostic."""
import numpy as np


def pairing_indices():
    return np.array([[[0, 1], [2, 3]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]], dtype='int32')


def select_pairings(contrast):
    patterns = pairing_indices()
    differences = contrast[:, patterns[:, :, 0]] - contrast[:, patterns[:, :, 1]]
    score = (differences ** 2).mean(axis=2)
    return np.argmin(score, axis=1), np.argmax(score, axis=1), score


def expected_random_loss(losses):
    return np.asarray(losses).mean(axis=-1)


def fixed_effect_fit(x, y, weights, blocks):
    x, y, weights = np.asarray(x, dtype='float64'), np.asarray(y, dtype='float64'), np.asarray(weights, dtype='float64')
    _, b = np.unique(blocks, return_inverse=True)
    total = np.bincount(b, weights=weights)
    mx = np.stack([np.bincount(b, weights=weights * x[:, k]) / np.maximum(total, 1e-20) for k in range(x.shape[1])], axis=1)
    my = np.bincount(b, weights=weights * y) / np.maximum(total, 1e-20)
    dx, dy = x - mx[b], y - my[b]
    root = np.sqrt(weights)
    return np.linalg.lstsq(dx * root[:, None], dy * root, rcond=1e-10)[0]
