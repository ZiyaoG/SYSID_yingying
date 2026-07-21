"""LSE confidence regions: Abbasi-Yadkori (2011) ellipsoid vs Simchowitz (2023) diameter.

Port of ``figure_1.m`` (``least_square`` + ``new_LSE_diam_fxn``).

Note: ``figure_1.m`` uses ``lambda = 0.1`` with regressors ``z = [x, u]`` of order
O(1).  The pendulum stores ``phi_dt = dt * phi`` (``||z|| ~ 0.01``), so the same
``lambda`` overwhelms ``Z^T Z`` and shrinks ``theta_hat`` toward zero.  Use a small
ridge (``1e-8``, as in ``pendulum_LSE_active.ipynb`` / ``pendulum_RLS``) instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from pendulum_RLS import fit_theta_from_gram

DEFAULT_LSE_RIDGE = 1e-8


@dataclass(frozen=True)
class LSEConfidenceRegion:
    """Plottable LSE confidence set (Abbasi ellipsoid or Simchowitz ball)."""

    center: np.ndarray
    M: np.ndarray
    beta: float
    diameter: float


def abbasi_lse(
    z: np.ndarray,
    y: np.ndarray,
    *,
    lam: float,
    L: float,
    S: float,
    delta: float,
    n_out: int = 1,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Regularized LS + Abbasi-Yadkori Theorem 1 ellipsoid (``least_square`` in figure_1.m)."""
    z_arr = np.asarray(z, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = min(z_arr.shape[0], y_arr.shape[0])
    if n == 0:
        d = z_arr.shape[1] if z_arr.ndim == 2 else 2
        V = lam * np.eye(d)
        return np.zeros(d), V, float("inf"), float("inf")

    z_arr = z_arr[:n]
    y_arr = y_arr[:n]
    d = z_arr.shape[1]
    gram = z_arr.T @ z_arr
    phiTy = z_arr.T @ y_arr
    V = lam * np.eye(d) + gram
    theta_hat = fit_theta_from_gram(gram, phiTy, ridge=lam)

    det_ratio = np.sqrt(np.linalg.det(V)) / np.sqrt(np.linalg.det(lam * np.eye(d)))
    beta_inner = np.sqrt(max(2.0 * np.log(max(det_ratio / delta, 1e-300)), 0.0)) * L * n_out
    beta = float((beta_inner + np.sqrt(lam) * S) ** 2)

    eigvals = np.linalg.eigvalsh(V)
    eigvals = np.maximum(eigvals, 1e-12)
    semi_axes = np.sqrt(beta / eigvals)
    diameter = float(2.0 * np.max(semi_axes))
    return theta_hat, V, beta, diameter


def simchowitz_lse_diameter(
    z: np.ndarray,
    y: np.ndarray,
    *,
    delta: float,
    sigma: float = 1.0,
    n_x: int = 1,
    n_z: int = 2,
) -> tuple[np.ndarray, float]:
    """Simchowitz Lemma E.3 diameter (``new_LSE_diam_fxn`` in figure_1.m)."""
    z_arr = np.asarray(z, dtype=float)
    y_arr = np.asarray(y, dtype=float).ravel()
    n = min(z_arr.shape[0], y_arr.shape[0])
    if n == 0:
        return np.zeros(n_z, dtype=float), float("inf")

    z_arr = z_arr[:n]
    y_arr = y_arr[:n]
    gamma = z_arr.T @ z_arr
    theta_hat = y_arr @ z_arr @ np.linalg.pinv(gamma)

    alpha_list, U = np.linalg.eigh(gamma)

    diam_options = []
    for p in range(1, n_z):
        P0 = np.diag(np.r_[np.ones(p), np.zeros(n_z - p)])
        P_matrix = U @ P0 @ U.T
        Q, _ = np.linalg.qr(P_matrix)

        lam1 = float(np.min(alpha_list[:p]))
        lam2 = float(np.min(alpha_list[p:]))
        lam1 = max(lam1, 1e-12)
        lam2 = max(lam2, 1e-12)

        kappa1 = max(float(Q[:, i].T @ gamma @ Q[:, i] / lam1) for i in range(p))
        kappa2 = max(
            float(Q[:, i + p].T @ gamma @ Q[:, i + p] / lam2)
            for i in range(n_z - p)
        )

        diam_sq = (
            12.0 * n_x * p * kappa1 * np.log(3.0 * n_x * n_z * kappa1 / delta) / lam1
            + 48.0 * n_x * (n_z - p) * kappa2 * np.log(3.0 * n_x * n_z * kappa2 / delta)
            / lam2
        )
        diam_options.append(np.sqrt(max(diam_sq, 0.0)) * sigma)

    return theta_hat, float(min(diam_options) if diam_options else float("inf"))


def lse_confidence_region(
    z: np.ndarray,
    y: np.ndarray,
    *,
    lam: float = DEFAULT_LSE_RIDGE,
    L: float,
    S: float,
    delta: float = 0.1,
    sigma: float = 1.0,
    n_out: int = 1,
) -> LSEConfidenceRegion:
    """Pick Abbasi ellipsoid or Simchowitz ball by smaller diameter (figure_1.m logic)."""
    theta_a, V, beta_a, diam_a = abbasi_lse(
        z, y, lam=lam, L=L, S=S, delta=delta, n_out=n_out
    )
    theta_s, diam_s = simchowitz_lse_diameter(
        z, y, delta=delta, sigma=sigma, n_x=n_out, n_z=z.shape[1] if z.ndim == 2 else 2
    )

    if diam_s < diam_a:
        d = theta_a.shape[0]
        return LSEConfidenceRegion(
            center=theta_s,
            M=np.eye(d),
            beta=diam_s**2,
            diameter=diam_s,
        )

    return LSEConfidenceRegion(
        center=theta_a,
        M=V,
        beta=beta_a,
        diameter=diam_a,
    )


def confidence_boundary_points(
    center: Sequence[float],
    M: np.ndarray,
    beta: float,
    *,
    n_pts: int = 200,
) -> np.ndarray:
    """Boundary of {theta : (theta-center)^T M (theta-center) <= beta}."""
    center_arr = np.asarray(center, dtype=float).ravel()
    M_arr = np.asarray(M, dtype=float)
    beta_val = float(beta)
    if not np.isfinite(beta_val) or beta_val <= 0:
        return center_arr.reshape(1, -1)

    w, V = np.linalg.eigh(0.5 * (M_arr + M_arr.T))
    w = np.maximum(w, 1e-12)
    scale = np.sqrt(beta_val / w)
    angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=True)
    unit = np.stack([np.cos(angles), np.sin(angles)], axis=0)
    pts = center_arr[:, None] + V @ (scale[:, None] * unit)
    return pts.T
