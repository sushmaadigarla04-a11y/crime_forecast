"""k-Shape time-series clustering (Paparrizos & Gravano, SIGMOD 2015) in pure NumPy.

* distance: shape-based distance SBD(x, y) = 1 - max_w NCC_w(x, y), computed with FFT
* centroid: shape extraction = leading eigenvector of Q^T S Q over the aligned members
* assignment / refinement iterate until labels stop changing

The interface (fit, predict, fit_predict, cluster_centers_, labels_, inertia_)
matches tslearn.clustering.KShape.
"""
from __future__ import annotations

import numpy as np


def zscore(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / np.where(sd == 0, 1, sd)


def _ncc(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation for every (series, centroid) pair and every shift.

    Returns array (n, k, 2m-1); index i corresponds to shift i - (m - 1).
    """
    m = X.shape[1]
    L = 1 << int(np.ceil(np.log2(2 * m - 1)))
    FX = np.fft.rfft(X, L)
    FC = np.fft.rfft(C, L)
    cc = np.fft.irfft(FX[:, None, :] * np.conj(FC[None, :, :]), L)
    cc = np.concatenate([cc[..., -(m - 1):], cc[..., :m]], axis=-1)
    den = np.linalg.norm(X, axis=1)[:, None] * np.linalg.norm(C, axis=1)[None, :]
    return cc / np.where(den == 0, np.inf, den)[..., None]


def sbd(X: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape-based distance (n, k) and the optimal shift for each pair."""
    ncc = _ncc(X, C)
    idx = ncc.argmax(axis=-1)
    dist = 1 - np.take_along_axis(ncc, idx[..., None], axis=-1)[..., 0]
    return dist, idx - (X.shape[1] - 1)


def _shift(y: np.ndarray, s: int) -> np.ndarray:
    if s == 0:
        return y
    out = np.zeros_like(y)
    if s > 0:
        out[s:] = y[:-s]
    else:
        out[:s] = y[-s:]
    return out


def _extract_shape(members: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    m = members.shape[1]
    if not np.any(centroid):
        aligned = members
    else:
        # shift = argmax index - (m-1) aligns `centroid` to each member; apply the inverse to the member
        _, shifts = sbd(members, centroid[None, :])
        aligned = np.stack([_shift(x, -int(s)) for x, s in zip(members, shifts[:, 0])])
    aligned = zscore(aligned)
    S = aligned.T @ aligned
    Q = np.eye(m) - np.full((m, m), 1.0 / m)
    M = Q.T @ S @ Q
    _, vecs = np.linalg.eigh(M)
    c = vecs[:, -1]
    # eigenvectors have arbitrary sign: keep the one closer to the members
    if np.linalg.norm(aligned - c) > np.linalg.norm(aligned + c):
        c = -c
    return zscore(c[None, :])[0]


class KShape:
    def __init__(self, n_clusters: int = 3, max_iter: int = 100, n_init: int = 1,
                 random_state: int | None = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def _single_run(self, X: np.ndarray, rng: np.random.Generator):
        n, m = X.shape
        k = self.n_clusters
        labels = rng.integers(0, k, n)
        C = np.zeros((k, m))
        for _ in range(self.max_iter):
            old = labels.copy()
            for j in range(k):
                members = X[labels == j]
                if len(members) == 0:                      # re-seed an empty cluster
                    members = X[[rng.integers(n)]]
                C[j] = _extract_shape(members, C[j])
            dist, _ = sbd(X, C)
            labels = dist.argmin(axis=1)
            if np.array_equal(labels, old):
                break
        dist, _ = sbd(X, C)
        inertia = float(dist[np.arange(n), labels].sum() / n)
        return labels, C.copy(), inertia

    def fit(self, X: np.ndarray) -> "KShape":
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            X = X[..., 0]
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            res = self._single_run(X, rng)
            if best is None or res[2] < best[2]:
                best = res
        self.labels_, self.cluster_centers_, self.inertia_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            X = X[..., 0]
        return sbd(zscore(X), self.cluster_centers_)[0].argmin(axis=1)

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).labels_
