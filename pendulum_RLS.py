"""
Recursive least squares for pendulum regression.

Model (same as ``lse_pend.run_lst``):  y_t ≈ theta^T phi_t
with normal equations  Gram = sum_t phi_t phi_t^T,  phiTy = sum_t phi_t y_t.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np

from lse_pend import error_theta_l2_rel

ArrayLike = Union[np.ndarray, Sequence[float]]


def fit_theta_from_gram(
    gram: np.ndarray,
    phiTy: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Minimum-norm least squares via normal equations.

    Uses ``pinv(Gram) @ (Phi^T y)`` so results match ``lse_pend.run_lst`` (which
    applies ``pinv(Phi) @ y``) even when Gram is rank-deficient.
    """
    G = np.asarray(gram, dtype=float) + ridge * np.eye(gram.shape[0])
    b = np.asarray(phiTy, dtype=float).ravel()
    return np.linalg.pinv(G) @ b


def p_matrix_from_gram(
    gram: np.ndarray,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Return the RLS covariance shape matrix P = (Phi^T Phi + ridge I)^(-1).

    Multiplying P by an observation-noise variance estimate gives the usual
    parameter covariance approximation. Use ridge > 0 before the design matrix
    is full rank; with ridge = 0 this returns the Moore-Penrose pseudoinverse.
    """
    G = np.asarray(gram, dtype=float) + ridge * np.eye(gram.shape[0])
    return np.linalg.pinv(G)


@dataclass
class RecursiveLS:
    """Online accumulation of Phi^T Phi and Phi^T y."""

    n_features: int = 2
    ridge: float = 0.0
    gram: np.ndarray = field(init=False)
    phiTy: np.ndarray = field(init=False)
    n_samples: int = 0

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.gram = np.zeros((self.n_features, self.n_features), dtype=float)
        self.phiTy = np.zeros(self.n_features, dtype=float)
        self.n_samples = 0

    def update(self, phi: ArrayLike, y: float) -> None:
        p = np.asarray(phi, dtype=float).ravel()
        if p.shape[0] != self.n_features:
            raise ValueError(
                f"phi has length {p.shape[0]}, expected n_features={self.n_features}"
            )
        self.gram += np.outer(p, p)
        self.phiTy += p * float(y)
        self.n_samples += 1

    def theta_hat(self) -> np.ndarray:
        if self.n_samples == 0:
            return np.zeros(self.n_features, dtype=float)
        return fit_theta_from_gram(self.gram, self.phiTy, ridge=self.ridge)

    def p_matrix(self) -> np.ndarray:
        return p_matrix_from_gram(self.gram, ridge=self.ridge)


def rls_theta_sequence(
    y: Sequence[float],
    phi_rows: np.ndarray,
    *,
    ridge: float = 0.0,
) -> np.ndarray:
    """
    Run RLS through a trajectory; return theta_hat after each (phi, y) pair.

    Returns shape (n_samples, n_features).
    """
    y_arr = np.asarray(y, dtype=float).ravel()
    phi = np.asarray(phi_rows, dtype=float)
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    n = min(len(y_arr), phi.shape[0])
    d = phi.shape[1]
    out = np.zeros((n, d), dtype=float)
    rls = RecursiveLS(n_features=d, ridge=ridge)
    for t in range(n):
        rls.update(phi[t], y_arr[t])
        out[t] = rls.theta_hat()
    return out


def rls_p_matrix_sequence(
    phi_rows: np.ndarray,
    *,
    ridge: float = 1e-8,
    update_every: Optional[int] = None,
) -> np.ndarray:
    """
    Return P_t = (Phi_t^T Phi_t + ridge I)^(-1) after each sample.

    If ``update_every`` is set, P is refreshed every that many samples and
    forward-filled between refreshes, matching ``rls_errors_sequence``.
    """
    phi = np.asarray(phi_rows, dtype=float)
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    n = phi.shape[0]
    if n == 0:
        return np.zeros((0, 0, 0), dtype=float)

    d = phi.shape[1]
    out = np.zeros((n, d, d), dtype=float)
    gram = np.zeros((d, d), dtype=float)
    batch_start = 0
    step = max(1, int(update_every)) if update_every is not None else 1
    last_p = p_matrix_from_gram(gram, ridge=ridge)

    for t in range(n):
        if (t + 1 - batch_start) >= step or t == n - 1:
            for s in range(batch_start, t + 1):
                gram += np.outer(phi[s], phi[s])
            batch_start = t + 1
            last_p = p_matrix_from_gram(gram, ridge=ridge)
        out[t] = last_p
    return out


def rls_uncertainty_sequence(
    phi_rows: np.ndarray,
    *,
    ridge: float = 1e-8,
    update_every: Optional[int] = None,
    metric: str = "sqrt_trace",
) -> np.ndarray:
    """
    Scalar uncertainty curve derived from the RLS P matrix.

    ``sqrt_trace`` is proportional to the root-sum marginal standard error when
    the observation-noise variance is fixed across methods.
    """
    p_seq = rls_p_matrix_sequence(
        phi_rows,
        ridge=ridge,
        update_every=update_every,
    )
    if p_seq.size == 0:
        return np.array([], dtype=float)
    if metric == "sqrt_trace":
        return np.sqrt(np.trace(p_seq, axis1=1, axis2=2))
    if metric == "max_std":
        return np.sqrt(np.max(np.diagonal(p_seq, axis1=1, axis2=2), axis=1))
    if metric == "logdet":
        signs, logdets = np.linalg.slogdet(p_seq)
        return np.where(signs > 0, logdets, np.nan)
    raise ValueError("metric must be one of {'sqrt_trace', 'max_std', 'logdet'}")


def rls_errors_sequence(
    y: Sequence[float],
    phi_rows: np.ndarray,
    theta_star: np.ndarray,
    *,
    ridge: float = 0.0,
    update_every: Optional[int] = None,
) -> np.ndarray:
    """
    Relative L2 error ||theta_hat - theta*|| / ||theta*|| vs sample index.

    If ``update_every`` is set, RLS is refreshed every that many samples (episode
    cadence); between updates the last error is held constant.
    """
    if update_every is not None and update_every > 1:
        return rls_errors_sequence_episode(
            y, phi_rows, theta_star, update_every=update_every, ridge=ridge
        )
    thetas = rls_theta_sequence(y, phi_rows, ridge=ridge)
    return np.array(
        [error_theta_l2_rel(thetas[t], theta_star) for t in range(thetas.shape[0])],
        dtype=float,
    )


def rls_errors_sequence_episode(
    y: Sequence[float],
    phi_rows: np.ndarray,
    theta_star: np.ndarray,
    *,
    update_every: int,
    ridge: float = 0.0,
) -> np.ndarray:
    """RLS batched every ``update_every`` samples; forward-fill errors between batches."""
    y_arr = np.asarray(y, dtype=float).ravel()
    phi = np.asarray(phi_rows, dtype=float)
    if phi.ndim == 1:
        phi = phi.reshape(1, -1)
    n = min(len(y_arr), phi.shape[0])
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out

    rls = RecursiveLS(n_features=phi.shape[1], ridge=ridge)
    last_err = np.nan
    batch_start = 0
    step = max(1, int(update_every))

    for t in range(n):
        if (t + 1 - batch_start) >= step or t == n - 1:
            for s in range(batch_start, t + 1):
                rls.update(phi[s], y_arr[s])
            batch_start = t + 1
            last_err = error_theta_l2_rel(rls.theta_hat(), theta_star)
        out[t] = last_err

    return out


def mean_std_error_curves(
    error_seqs: List[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise mean and std across trajectories (equal length)."""
    stack = np.stack(error_seqs, axis=0)
    return np.mean(stack, axis=0), np.std(stack, axis=0)
