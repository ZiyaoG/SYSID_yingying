"""Recursive ellipsoid SME (Fogel & Huang, 1982, Algorithms 1 and 2).

The estimator keeps an ellipsoid

    E_t = {theta : (theta - c_t)^T P_t^{-1} (theta - c_t) <= 1}

and updates it with the minimum-volume or minimum-trace outer bound of the
intersection with each new measurement strip. Measurements that do not shrink
the selected size measure are skipped (q_k = 0).
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from set_membership_lin_prog_pend import uncertainty_set_coordinate_bounds


@dataclass
class EllipsoidSMECurveSummary:
    """Aggregated ellipsoid-SME width curves and first-hit statistics."""

    width_norm_seqs: List[np.ndarray]
    reach_T: List[Optional[int]]
    reach_metric: List[float]
    mean_width_norm: np.ndarray
    std_width_norm: np.ndarray
    wall_s: float
    wall_s_by_traj: List[float]
    reach_T_mean: float


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)


def _nearest_spd(matrix: np.ndarray, min_eig: float) -> np.ndarray:
    values, vectors = np.linalg.eigh(_symmetrize(matrix))
    max_eig = float(np.max(values)) if values.size else 1.0
    eig_floor = max(float(min_eig), 100.0 * np.finfo(float).eps * max(max_eig, 1.0))
    values = np.maximum(values, eig_floor)
    return vectors @ np.diag(values) @ vectors.T


def final_finite_value(seq: Sequence[float]) -> float:
    """Last finite value in a curve, or nan if no finite entry exists."""
    values = np.asarray(seq, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else float("nan")


def summarize_sme_stack(
    width_norm_seqs: Sequence[np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise nan mean/std for ellipsoid SME width curves."""
    if not width_norm_seqs:
        nan_curve = np.full(int(horizon), np.nan, dtype=float)
        return nan_curve, nan_curve.copy()

    stack = np.stack(width_norm_seqs, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)


def minimum_volume_q_k(
    n: int,
    G: float,
    eps_sq: float,
) -> float:
    """Optimal ``q_k^v`` from Fogel & Huang (1982), eq. (20)-(21)."""
    G = float(G)
    eps_sq = float(eps_sq)
    if G <= 0.0 or not np.isfinite(G) or not np.isfinite(eps_sq):
        return 0.0

    a1 = (n - 1) * G *G
    a2 = G * (2 * n - 1 - G + eps_sq)
    a3 = n * (1.0 - eps_sq) - G
    disc = a2 * a2 - 4.0 * a1 * a3
    if disc < 0.0:
        return 0.0
    sqrt_disc = float(np.sqrt(disc))
    if -a2 + sqrt_disc <= 0.0:
        return 0.0
    return float((-a2 + sqrt_disc) / (2.0 * a1))


def _positive_real_roots(coefficients: Sequence[float], *, tol: float = 1e-12) -> np.ndarray:
    """Return positive real roots after dropping numerically zero leading terms."""
    coeffs = np.asarray(coefficients, dtype=float)
    finite_scale = np.max(np.abs(coeffs[np.isfinite(coeffs)])) if np.any(np.isfinite(coeffs)) else 0.0
    if finite_scale <= 0.0:
        return np.empty(0, dtype=float)

    keep_start = 0
    while keep_start < coeffs.size and abs(coeffs[keep_start]) <= tol * finite_scale:
        keep_start += 1
    coeffs = coeffs[keep_start:]
    if coeffs.size <= 1:
        return np.empty(0, dtype=float)

    roots = np.roots(coeffs)
    real_roots = roots[np.isfinite(roots) & (np.abs(roots.imag) <= tol * np.maximum(1.0, np.abs(roots.real)))].real
    return np.sort(real_roots[real_roots > tol])


def minimum_trace_q_k(
    trace_old: float,
    G: float,
    gamma: float,
    eps_sq: float,
) -> float:
    """Optimal ``q_k^T`` from Fogel & Huang (1982), Algorithm 2.

    This solves the cubic stationarity condition for
    ``Tr(P_k) = z_k [Tr(P_{k-1}) - q gamma / (1 + q G)]`` and selects the
    positive root giving the smallest trace. ``gamma`` is eq. (23).
    """
    trace_old = float(trace_old)
    G = float(G)
    gamma = float(gamma)
    eps_sq = float(eps_sq)
    if (
        trace_old <= 0.0
        or G <= 0.0
        or gamma <= 0.0
        or not np.all(np.isfinite([trace_old, G, gamma, eps_sq]))
    ):
        return 0.0

    coefficients = [
        G**2 * (gamma - G * trace_old),
        3.0 * G * (gamma - G * trace_old),
        eps_sq * G * trace_old
        - 2.0 * eps_sq * gamma
        - 3.0 * G * trace_old
        + G * gamma
        + 2.0 * gamma,
        trace_old * (eps_sq - 1.0) + gamma,
    ]

    best_q = 0.0
    best_trace = trace_old
    for q in _positive_real_roots(coefficients):
        denom = 1.0 + q * G
        if denom <= 0.0:
            continue
        zeta = 1.0 + q - q * eps_sq / denom
        trace_candidate = zeta * (trace_old - q * gamma / denom)
        if np.isfinite(trace_candidate) and trace_candidate < best_trace:
            best_trace = float(trace_candidate)
            best_q = float(q)
    return best_q


def _strip_margin_along_normal(shape: np.ndarray, direction: np.ndarray) -> float:
    """Max |a^T (theta - c)| over the ellipsoid; equals sqrt(a^T P a)."""
    direction = np.asarray(direction, dtype=float).reshape(-1)
    return float(np.sqrt(max(direction @ shape @ direction, 0.0)))


def fogel_huang_update(
    center: np.ndarray,
    shape: np.ndarray,
    direction: np.ndarray,
    measurement: float,
    *,
    noise_scale: float,
    objective: str = "volume",
    min_eig: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """One Fogel & Huang ellipsoid update for a single strip.

    The strip is ``|measurement - direction^T theta| <= 1 / noise_scale``,
    matching ``|delta - phi^T theta| <= w_max`` with ``noise_scale = 1 / w_max``.
    """
    center = np.asarray(center, dtype=float).reshape(-1)
    shape = _nearest_spd(shape, min_eig)
    direction = np.asarray(direction, dtype=float).reshape(-1)
    n = int(center.size)
    r = float(noise_scale)
    if r <= 0.0:
        raise ValueError("noise_scale must be positive")
    objective_normalized = str(objective).lower()
    if objective_normalized not in {"volume", "trace"}:
        raise ValueError('objective must be "volume" or "trace"')

    residual = float(measurement - direction @ center)
    scaled_residual = r * residual
    scaled_direction = r * direction
    scaled_residual_sq = scaled_residual * scaled_residual
    G = float(scaled_direction @ shape @ scaled_direction)
    gamma = float(scaled_direction @ shape @ shape @ scaled_direction)
    margin = _strip_margin_along_normal(shape, direction)

    scaled_margin = r * margin
    if abs(scaled_residual) > 1.0 + scaled_margin + 1e-12:
        return center, shape, 0.0, "empty"

    if abs(scaled_residual) + scaled_margin <= 1.0 + 1e-12:
        return center, shape, 0.0, "optimal"

    if objective_normalized == "volume":
        q = minimum_volume_q_k(n, G, scaled_residual_sq)
    else:
        q = minimum_trace_q_k(float(np.trace(shape)), G, gamma, scaled_residual_sq)
    if q <= 0.0:
        return center, shape, 0.0, "optimal"

    sign_old, logdet_old = np.linalg.slogdet(shape)
    if sign_old <= 0 or not np.isfinite(logdet_old):
        return center, shape, 0.0, "optimal"
    trace_old = float(np.trace(shape))
    if trace_old <= 0.0 or not np.isfinite(trace_old):
        return center, shape, 0.0, "optimal"

    Z = shape - (
        q
        * (shape @ np.outer(scaled_direction, scaled_direction) @ shape)
        / (1.0 + q * G)
    )
    zeta = 1.0 + q - q * scaled_residual_sq / (1.0 + q * G)
    if zeta <= 0.0 or not np.isfinite(zeta):
        return center, shape, 0.0, "optimal"

    candidate_shape = _nearest_spd(zeta * Z, min_eig)
    candidate_center = center + q * Z @ scaled_direction * scaled_residual

    sign_new, logdet_new = np.linalg.slogdet(candidate_shape)
    if sign_new <= 0 or not np.isfinite(logdet_new):
        return center, shape, 0.0, "optimal"
    trace_new = float(np.trace(candidate_shape))
    if trace_new <= 0.0 or not np.isfinite(trace_new):
        return center, shape, 0.0, "optimal"

    if objective_normalized == "volume" and logdet_new >= logdet_old - 1e-12:
        return center, shape, 0.0, "optimal"
    if objective_normalized == "trace" and trace_new >= trace_old - 1e-12:
        return center, shape, 0.0, "optimal"

    return candidate_center, candidate_shape, q, "optimal"


def fogel_huang_minimum_volume_update(
    center: np.ndarray,
    shape: np.ndarray,
    direction: np.ndarray,
    measurement: float,
    *,
    noise_scale: float,
    min_eig: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    """Backward-compatible minimum-volume Fogel-Huang update."""
    return fogel_huang_update(
        center,
        shape,
        direction,
        measurement,
        noise_scale=noise_scale,
        objective="volume",
        min_eig=min_eig,
    )


@dataclass
class RecursiveEllipsoidSME:
    """Stateful Fogel & Huang ellipsoid SME."""

    center: np.ndarray
    shape: np.ndarray
    w_max: float
    initialized: bool = False
    status: str = "empty"
    min_eig: float = 1e-12
    solver: str = "linprog"
    objective: str = "volume"

    @property
    def noise_scale(self) -> float:
        return 1.0 / float(self.w_max)

    @classmethod
    def empty(
        cls,
        w_max: float,
        min_eig: float = 1e-12,
        *,
        solver: str = "linprog",
        objective: str = "volume",
    ) -> "RecursiveEllipsoidSME":
        return cls(
            center=np.full(2, np.nan, dtype=float),
            shape=np.full((2, 2), np.nan, dtype=float),
            w_max=float(w_max),
            min_eig=float(min_eig),
            solver=solver,
            objective=str(objective).lower(),
        )

    def initialize(
        self,
        delta_s: Sequence[float],
        phi_s_u: np.ndarray,
    ) -> str:
        """Initialize from exact coordinate bounds for the constraints seen so far."""
        lower, upper, status = uncertainty_set_coordinate_bounds(
            delta_s,
            np.asarray(phi_s_u, dtype=float),
            self.w_max,
            solver=self.solver,
        )
        if status != "optimal" or not np.all(np.isfinite(lower + upper)):
            self.status = status
            return status

        half_width = 0.5 * np.maximum(upper - lower, 0.0)
        self.center = 0.5 * (lower + upper)
        self.shape = np.diag(2.0 * np.maximum(half_width, np.sqrt(self.min_eig)) ** 2)
        self.shape = _nearest_spd(self.shape, self.min_eig)
        self.initialized = True
        self.status = "optimal"
        return self.status

    def update(self, phi: Sequence[float], delta: float) -> str:
        """Apply one Fogel & Huang ellipsoid update."""
        if not self.initialized:
            raise RuntimeError("RecursiveEllipsoidSME must be initialized before update")

        center, shape, q, status = fogel_huang_update(
            self.center,
            self.shape,
            np.asarray(phi, dtype=float).reshape(2,),
            float(delta),
            noise_scale=self.noise_scale,
            objective=self.objective,
            min_eig=self.min_eig,
        )
        if status == "optimal" and q > 0.0:
            self.center = center
            self.shape = shape
        self.status = status
        return self.status

    def coordinate_bounds(self) -> tuple[np.ndarray, np.ndarray, str]:
        if not self.initialized or self.status not in {"optimal", "empty"}:
            nan_bounds = np.full(2, np.nan, dtype=float)
            return nan_bounds, nan_bounds, self.status
        if self.status == "empty":
            nan_bounds = np.full(2, np.nan, dtype=float)
            return nan_bounds, nan_bounds, self.status

        diag = np.maximum(np.diag(_nearest_spd(self.shape, self.min_eig)), 0.0)
        radius = np.sqrt(diag)
        return self.center - radius, self.center + radius, "optimal"

    def coordinate_width(self) -> float:
        lower, upper, status = self.coordinate_bounds()
        if status != "optimal":
            return float("nan")
        return float(np.max(upper - lower))


def ellipsoid_sme_coordinate_width_sequence(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
    *,
    min_T: int = 2,
    theta_star: Optional[np.ndarray] = None,
    tol: Optional[float] = None,
    max_T: Optional[int] = None,
    stop_at_tol: bool = False,
    solver: str = "linprog",
    objective: str = "volume",
) -> tuple[np.ndarray, Optional[int], float, RecursiveEllipsoidSME]:
    """Return the normalized Fogel-Huang ellipsoid-SME width curve."""
    phi = np.asarray(phi_s_u, dtype=float)
    delta = np.asarray(delta_s, dtype=float)
    n = min(delta.shape[0], phi.shape[0])
    if max_T is not None:
        n = min(n, int(max_T))

    out = np.full(n, np.nan, dtype=float)
    estimator = RecursiveEllipsoidSME.empty(
        w_max=w_max,
        solver=solver,
        objective=objective,
    )
    theta_norm = (
        float(np.linalg.norm(np.asarray(theta_star, dtype=float)) + 1e-12)
        if theta_star is not None
        else 1.0
    )
    reach_T: Optional[int] = None
    reach_metric = float("nan")

    for T in range(1, n + 1):
        if T < min_T:
            continue

        if estimator.initialized:
            estimator.update(phi[T - 1], float(delta[T - 1]))
        else:
            estimator.initialize(delta[:T], phi[:T])

        if estimator.status not in {"optimal", "empty"}:
            continue
        if estimator.status == "empty":
            continue

        metric = estimator.coordinate_width() / theta_norm
        out[T - 1] = metric
        if reach_T is None and tol is not None and np.isfinite(metric) and metric <= tol:
            reach_T = T
            reach_metric = metric
            if stop_at_tol:
                break

    return out, reach_T, reach_metric, estimator


def run_ellipsoid_sme_curves(
    delta_s_list: Sequence[Sequence[float]],
    phi_s_u_list: Sequence[np.ndarray],
    *,
    n_epoch_sme: int,
    horizon: int,
    print_horizon_label: str,
    w_max: float,
    min_T: int,
    theta_star: np.ndarray,
    tol: Optional[float],
    stop_at_tol: bool = False,
    solver: str = "linprog",
    objective: str = "volume",
) -> EllipsoidSMECurveSummary:
    """Run Fogel-Huang ellipsoid-SME curves over multiple trajectories."""
    t_start = time.perf_counter()
    width_norm_seqs: List[np.ndarray] = []
    reach_T: List[Optional[int]] = []
    reach_metric: List[float] = []
    wall_s_by_traj: List[float] = []

    objective_label = "volume" if str(objective).lower() == "volume" else "trace"
    label = f"Fogel-Huang ellipsoid SME ({objective_label})"

    print(f"{label} until {print_horizon_label}...")
    for e in range(int(n_epoch_sme)):
        print(f"  trajectory {e + 1} / {n_epoch_sme}", end="")
        t_traj = time.perf_counter()
        width_norm_e, reach_T_e, reach_metric_e, _estimator = (
            ellipsoid_sme_coordinate_width_sequence(
                delta_s_list[e],
                phi_s_u_list[e],
                w_max,
                min_T=min_T,
                theta_star=theta_star,
                tol=tol,
                max_T=horizon,
                stop_at_tol=stop_at_tol,
                solver=solver,
                objective=objective_label,
            )
        )
        wall_s_by_traj.append(time.perf_counter() - t_traj)
        width_norm_seqs.append(width_norm_e)
        reach_T.append(reach_T_e)
        reach_metric.append(reach_metric_e)

        final_metric_e = final_finite_value(width_norm_e)
        if reach_T_e is None:
            print(f"  -> not reached; final metric={final_metric_e:.2e}")
        else:
            print(
                f"  -> first reached at T={reach_T_e}, "
                f"final metric={final_metric_e:.2e}"
            )

    mean_width_norm, std_width_norm = summarize_sme_stack(width_norm_seqs, horizon)
    wall_s = time.perf_counter() - t_start
    reached = [t for t in reach_T if t is not None]
    reach_T_mean = float(np.mean(reached)) if reached else float("nan")

    print(f"{label} done. wall (s): {wall_s:.2f}")
    if reached:
        print(
            f"  First reached in {len(reached)}/{n_epoch_sme} runs; "
            f"mean first-hit T = {reach_T_mean:.0f}"
        )
    else:
        print("  Threshold not reached")

    return EllipsoidSMECurveSummary(
        width_norm_seqs=width_norm_seqs,
        reach_T=reach_T,
        reach_metric=reach_metric,
        mean_width_norm=mean_width_norm,
        std_width_norm=std_width_norm,
        wall_s=wall_s,
        wall_s_by_traj=wall_s_by_traj,
        reach_T_mean=reach_T_mean,
    )
