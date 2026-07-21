"""SME polytope for pendulum regression in the $(\\theta_1, \\theta_2)$ plane."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection

from set_membership_non_analytic import find_interior_point_2d


def build_pendulum_sme_halfspaces(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    phi = np.asarray(phi_s_u, dtype=float)
    y = np.asarray(delta_s, dtype=float).ravel()
    n = min(len(y), phi.shape[0])
    w = float(w_max)
    Ab, AA, bb = [], [], []
    for t in range(n):
        p = phi[t]
        yt = float(y[t])
        AA.extend([p.tolist(), (-p).tolist()])
        bb.extend([w + yt, w - yt])
        Ab.extend([p.tolist() + [-(w + yt)], (-p).tolist() + [-(w - yt)]])
    return Ab, AA, bb


def sme_polytope_vertices(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
) -> Tuple[np.ndarray, str]:
    Ab, AA, bb = build_pendulum_sme_halfspaces(delta_s, phi_s_u, w_max)
    if not Ab:
        return np.zeros((0, 2), dtype=float), "empty"
    half_spaces = np.asarray(Ab, dtype=float)
    if half_spaces.shape[0] < 3:
        return _coordinate_box(AA, bb), "degenerate"
    interior = find_interior_point_2d(half_spaces, AA, bb)
    try:
        hs = HalfspaceIntersection(half_spaces, interior, incremental=True, qhull_options="QJ")
        return hs.intersections, "optimal"
    except Exception:
        return _coordinate_box(AA, bb), "coordinate_bounds"


def sme_directional_diameter(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
    *,
    n_dirs: int = 36,
) -> float:
    """Max width of the SME set over uniformly sampled directions."""
    _, AA, bb = build_pendulum_sme_halfspaces(delta_s, phi_s_u, w_max)
    if not AA:
        return float("nan")
    A = np.asarray(AA, dtype=float)
    b = np.asarray(bb, dtype=float)
    bounds = [(None, None), (None, None)]
    diam = 0.0
    for ang in np.linspace(0.0, np.pi, n_dirs, endpoint=False):
        d = np.array([np.cos(ang), np.sin(ang)], dtype=float)
        lo = linprog(d, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        hi = linprog(-d, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        if lo.success and hi.success:
            diam = max(diam, float(-hi.fun - lo.fun))
    return diam


def _coordinate_box(AA: Sequence[Sequence[float]], bb: Sequence[float]) -> np.ndarray:
    A = np.asarray(AA, dtype=float)
    b = np.asarray(bb, dtype=float)
    bounds = [(None, None), (None, None)]
    out = []
    for dim in (0, 1):
        c = np.zeros(2, dtype=float)
        c[dim] = 1.0
        lo = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        hi = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        out.append((float(lo.fun) if lo.success else -1.0, float(-hi.fun) if hi.success else 1.0))
    (a_lo, a_hi), (b_lo, b_hi) = out
    return np.array([[a_lo, b_lo], [a_hi, b_lo], [a_hi, b_hi], [a_lo, b_hi]], dtype=float)
