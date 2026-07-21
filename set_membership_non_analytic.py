"""Vanilla SME for the scalar non-analytic bump example."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection

from non_analytic_dynamics import phi_features


def _append_one_sided_constraints(
    Ab: list[list[float]],
    AA: list[list[float]],
    bb: list[float],
    phi: Sequence[float],
    upper_bound: float,
) -> None:
    phi_arr = np.asarray(phi, dtype=float).ravel()
    if float(np.linalg.norm(phi_arr)) < 1e-12:
        return
    AA.append(phi_arr.tolist())
    bb.append(float(upper_bound))
    Ab.append(phi_arr.tolist() + [-float(upper_bound)])


def _append_scalar_equation_constraints(
    Ab: list[list[float]],
    AA: list[list[float]],
    bb: list[float],
    phi: Sequence[float],
    y: float,
    w_max: float,
    *,
    min_norm: float = 1e-12,
) -> None:
    phi_arr = np.asarray(phi, dtype=float).ravel()
    if float(np.linalg.norm(phi_arr)) < min_norm:
        return
    AA.append(phi_arr.tolist())
    bb.append(float(w_max + y))
    AA.append((-phi_arr).tolist())
    bb.append(float(w_max - y))
    Ab.append(phi_arr.tolist() + [-(float(w_max) + float(y))])
    Ab.append((-phi_arr).tolist() + [-(float(w_max) - float(y))])


def build_non_analytic_halfspaces(
    x: Sequence[float],
    u: Sequence[float],
    x_next: Sequence[float],
    *,
    w_max: float,
    box_bounds: Optional[Sequence[tuple[float, float]]] = None,
) -> tuple[list[list[float]], list[list[float]], list[float]]:
    """Build LP / HalfspaceIntersection constraints for theta = [a, b]."""
    n = min(len(x), len(u), len(x_next))
    Ab: list[list[float]] = []
    AA: list[list[float]] = []
    bb: list[float] = []

    for t in range(n):
        phi = phi_features(float(x[t]), float(u[t]))
        _append_scalar_equation_constraints(Ab, AA, bb, phi, float(x_next[t]), w_max)

    if box_bounds is not None:
        for dim, (lo, hi) in enumerate(box_bounds):
            row = [0.0, 0.0]
            row[dim] = 1.0
            _append_one_sided_constraints(Ab, AA, bb, row, hi)
            row_neg = [0.0, 0.0]
            row_neg[dim] = -1.0
            _append_one_sided_constraints(Ab, AA, bb, row_neg, -lo)

    return Ab, AA, bb


def coordinate_bounds_2d(
    AA: Sequence[Sequence[float]],
    bb: Sequence[float],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Axis-aligned LP bounds for theta = [a, b]."""
    A = np.asarray(AA, dtype=float)
    b = np.asarray(bb, dtype=float)
    out: list[tuple[float, float]] = []
    for dim in (0, 1):
        c = np.zeros(2, dtype=float)
        c[dim] = 1.0
        lo = linprog(c, A_ub=A, b_ub=b, bounds=[(None, None)] * 2, method="highs")
        hi = linprog(-c, A_ub=A, b_ub=b, bounds=[(None, None)] * 2, method="highs")
        lo_val = float(lo.fun) if lo.success else -1.0
        hi_val = float(-hi.fun) if hi.success else 1.0
        out.append((lo_val, hi_val))
    return out[0], out[1]


def coordinate_polygon_2d(
    AA: Sequence[Sequence[float]],
    bb: Sequence[float],
) -> np.ndarray:
    """Rectangle corners from coordinate-wise LP bounds (fallback polygon)."""
    (a_lo, a_hi), (b_lo, b_hi) = coordinate_bounds_2d(AA, bb)
    return np.array(
        [
            [a_lo, b_lo],
            [a_hi, b_lo],
            [a_hi, b_hi],
            [a_lo, b_hi],
        ],
        dtype=float,
    )


def find_interior_point_2d(
    half_spaces: np.ndarray,
    AA: Sequence[Sequence[float]],
    bb: Sequence[float],
) -> np.ndarray:
    """Return a point strictly inside the 2D feasible set."""
    if half_spaces.shape[0] == 0:
        return np.zeros(2, dtype=float)

    (a_lo, a_hi), (b_lo, b_hi) = coordinate_bounds_2d(AA, bb)
    shrink_a = 1e-4 * max(a_hi - a_lo, 1.0)
    shrink_b = 1e-4 * max(b_hi - b_lo, 1.0)
    center = np.array(
        [
            0.5 * (a_lo + a_hi) + (shrink_a if a_hi - a_lo < 1e-6 else 0.0),
            0.5 * (b_lo + b_hi) + (shrink_b if b_hi - b_lo < 1e-6 else 0.0),
        ],
        dtype=float,
    )
    if a_hi - a_lo >= 1e-6:
        center[0] = min(max(center[0], a_lo + shrink_a), a_hi - shrink_a)
    if b_hi - b_lo >= 1e-6:
        center[1] = min(max(center[1], b_lo + shrink_b), b_hi - shrink_b)

    vals = half_spaces[:, 0] * center[0] + half_spaces[:, 1] * center[1] + half_spaces[:, 2]
    if np.all(vals < -1e-10):
        return center

    A = half_spaces[:, :2]
    b = -half_spaces[:, 2]
    norms = np.linalg.norm(A, axis=1)
    norms = np.where(norms < 1e-12, 1.0, norms)
    sol = linprog(
        np.array([0.0, 0.0, -1.0], dtype=float),
        A_ub=np.hstack([A, norms.reshape(-1, 1)]),
        b_ub=b,
        bounds=[(None, None), (None, None), (1e-8, None)],
        method="highs",
    )
    if sol.success and sol.x is not None:
        return np.asarray(sol.x[:2], dtype=float)
    return center


def run_set_membership_non_analytic(
    x: Sequence[float],
    u: Sequence[float],
    x_next: Sequence[float],
    *,
    w_max: float,
    box_bounds: Sequence[tuple[float, float]],
    verbose: bool = False,
) -> Tuple[np.ndarray, str]:
    """Return 2D vertices of the SME feasible set in the (a, b) plane."""
    Ab, AA, bb = build_non_analytic_halfspaces(
        x,
        u,
        x_next,
        w_max=w_max,
        box_bounds=box_bounds,
    )
    if verbose:
        print(f"SME (a,b) plane, T={len(x)}")

    if not Ab:
        return np.zeros((0, 2), dtype=float), "empty"

    half_spaces = np.asarray(Ab, dtype=float)
    if half_spaces.shape[0] < 3:
        return coordinate_polygon_2d(AA, bb), "degenerate"

    interior = find_interior_point_2d(half_spaces, AA, bb)
    try:
        hs = HalfspaceIntersection(half_spaces, interior, incremental=True, qhull_options="QJ")
        return hs.intersections, "optimal"
    except Exception:
        return coordinate_polygon_2d(AA, bb), "coordinate_bounds"


def polygon_plot_points(vertices: np.ndarray) -> np.ndarray:
    """Order 2D vertices for plotting; handle collinear/degenerate sets."""
    pts = np.asarray(vertices, dtype=float)
    if pts.size == 0:
        return np.zeros((0, 2), dtype=float)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    pts = pts[:, :2]

    if pts.shape[0] <= 2:
        return pts

    span = np.ptp(pts, axis=0)
    scale = max(float(np.max(np.abs(pts))), 1.0)
    tol = 1e-10 * scale

    if span[0] < tol and span[1] < tol:
        return pts[:1]

    if span[0] < tol:
        x = float(pts[0, 0])
        b_lo, b_hi = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        width = 0.01 * max(b_hi - b_lo, 1e-3)
        return np.array(
            [
                [x - width, b_lo],
                [x + width, b_lo],
                [x + width, b_hi],
                [x - width, b_hi],
            ],
            dtype=float,
        )

    if span[1] < tol:
        y = float(pts[0, 1])
        a_lo, a_hi = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        height = 0.01 * max(a_hi - a_lo, 1e-3)
        return np.array(
            [
                [a_lo, y - height],
                [a_hi, y - height],
                [a_hi, y + height],
                [a_lo, y + height],
            ],
            dtype=float,
        )

    try:
        hull = ConvexHull(pts)
        return pts[hull.vertices]
    except Exception:
        a_lo, a_hi = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
        b_lo, b_hi = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
        return np.array(
            [
                [a_lo, b_lo],
                [a_hi, b_lo],
                [a_hi, b_hi],
                [a_lo, b_hi],
            ],
            dtype=float,
        )


def plot_projected_polygon(
    ax,
    vertices: np.ndarray,
    color: str,
    alpha: float,
    label: str,
) -> None:
    """Fill a (possibly degenerate) 2D SME polygon."""
    poly = polygon_plot_points(vertices)
    if poly.shape[0] == 0:
        return
    if poly.shape[0] == 1:
        ax.scatter(poly[0, 0], poly[0, 1], color=color, s=60, label=label, zorder=3)
        return
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha, label=label)
