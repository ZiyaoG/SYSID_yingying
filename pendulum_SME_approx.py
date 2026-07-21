"""Recursive template-polytope SME for pendulum system identification.

The exact SME routine keeps every historical strip constraint and solves LPs
whose constraint count grows with time. This module keeps a fixed template

    P_t = {theta : A theta <= b_t}

and only updates the support values ``b_t`` when a new strip arrives. The update
LPs have 2 variables and ``n_template + 2`` inequalities, so the per-sample cost
is independent of the data horizon once the template has been initialized.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from set_membership_lin_prog_pend import sme_coordinate_width_at_T, solve_sme_lp


def regular_template_directions(n_sides: int = 4) -> np.ndarray:
    """Return outward normal directions for a regular 2D template polygon."""
    n = int(n_sides)
    if n < 4:
        raise ValueError("n_sides must be at least 4")
    if n % 2 != 0:
        raise ValueError("n_sides must be even so opposite support bounds exist")

    angles = 2.0 * np.pi * np.arange(n, dtype=float) / float(n)
    return np.column_stack((np.cos(angles), np.sin(angles)))


def strip_constraints(phi: Sequence[float], delta: float, w_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the two half-plane constraints induced by one SME measurement."""
    phi_arr = np.asarray(phi, dtype=float).reshape(2,)
    delta_val = float(delta)
    return (
        np.vstack((phi_arr, -phi_arr)),
        np.array([float(w_max) + delta_val, float(w_max) - delta_val], dtype=float),
    )


def sme_constraints_array(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized historical strip constraints, used only for initialization."""
    phi = np.asarray(phi_s_u, dtype=float)
    delta = np.asarray(delta_s, dtype=float)
    n = min(delta.shape[0], phi.shape[0])
    phi = phi[:n]
    delta = delta[:n]

    A_ub = np.empty((2 * n, 2), dtype=float)
    b_ub = np.empty(2 * n, dtype=float)
    A_ub[0::2] = phi
    b_ub[0::2] = float(w_max) + delta
    A_ub[1::2] = -phi
    b_ub[1::2] = float(w_max) - delta
    return A_ub, b_ub


def template_support_from_constraints(
    template_A: np.ndarray,
    constraint_A: np.ndarray,
    constraint_b: np.ndarray,
    *,
    solver: str = "linprog",
) -> tuple[np.ndarray, str]:
    """Project a feasible set onto the fixed template directions."""
    A_template = np.asarray(template_A, dtype=float)
    A_ub = np.asarray(constraint_A, dtype=float)
    b_ub = np.asarray(constraint_b, dtype=float)
    support = np.full(A_template.shape[0], np.nan, dtype=float)

    if A_ub.shape[0] == 0:
        return support, "empty"

    variable_bounds = [(None, None), (None, None)]
    for j, direction in enumerate(A_template):
        success, x, message = solve_sme_lp(
            -direction,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=variable_bounds,
            solver=solver,
        )
        if not success:
            return support, message
        value = float(direction @ x)
        if not np.isfinite(value):
            return support, "non-finite support"
        support[j] = value
    return support, "optimal"


def update_template_support(
    template_A: np.ndarray,
    template_b: np.ndarray,
    phi: Sequence[float],
    delta: float,
    w_max: float,
    *,
    solver: str = "linprog",
) -> tuple[np.ndarray, str]:
    """Update template supports after one new strip constraint."""
    A_template = np.asarray(template_A, dtype=float)
    b_template = np.asarray(template_b, dtype=float)
    strip_A, strip_b = strip_constraints(phi, delta, w_max)
    A_ub = np.vstack((A_template, strip_A))
    b_ub = np.concatenate((b_template, strip_b))
    return template_support_from_constraints(A_template, A_ub, b_ub, solver=solver)


def coordinate_bounds_from_template(
    template_A: np.ndarray,
    template_b: np.ndarray,
    *,
    solver: str = "linprog",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Compute coordinate min/max bounds of the template polytope."""
    A_ub = np.asarray(template_A, dtype=float)
    b_ub = np.asarray(template_b, dtype=float)
    lower = np.full(2, np.nan, dtype=float)
    upper = np.full(2, np.nan, dtype=float)

    if not np.all(np.isfinite(b_ub)):
        return lower, upper, "non-finite support"

    variable_bounds = [(None, None), (None, None)]
    for dim in range(2):
        c_min = np.zeros(2, dtype=float)
        c_min[dim] = 1.0
        success_min, x_min, message_min = solve_sme_lp(
            c_min,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=variable_bounds,
            solver=solver,
        )
        if not success_min:
            return lower, upper, message_min
        lower[dim] = float(x_min[dim])

        success_max, x_max, message_max = solve_sme_lp(
            -c_min,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=variable_bounds,
            solver=solver,
        )
        if not success_max:
            return lower, upper, message_max
        upper[dim] = float(x_max[dim])

    return lower, upper, "optimal"


def coordinate_width_from_template(
    template_A: np.ndarray,
    template_b: np.ndarray,
    *,
    solver: str = "linprog",
) -> float:
    """Largest coordinate width of the template polytope."""
    lower, upper, status = coordinate_bounds_from_template(
        template_A,
        template_b,
        solver=solver,
    )
    if status != "optimal":
        return float("nan")
    return float(np.max(upper - lower))


@dataclass
class RecursiveTemplateSME:
    """Stateful recursive SME approximation with fixed template directions."""

    template_A: np.ndarray
    template_b: np.ndarray
    w_max: float
    solver: str = "linprog"
    initialized: bool = False
    status: str = "empty"

    @classmethod
    def regular(
        cls,
        n_sides: int,
        w_max: float,
        *,
        solver: str = "linprog",
    ) -> "RecursiveTemplateSME":
        A = regular_template_directions(n_sides)
        return cls(
            template_A=A,
            template_b=np.full(A.shape[0], np.nan, dtype=float),
            w_max=float(w_max),
            solver=solver,
        )

    def initialize(
        self,
        delta_s: Sequence[float],
        phi_s_u: np.ndarray,
    ) -> str:
        """Initialize template supports from the constraints seen so far."""
        constraint_A, constraint_b = sme_constraints_array(delta_s, phi_s_u, self.w_max)
        support, status = template_support_from_constraints(
            self.template_A,
            constraint_A,
            constraint_b,
            solver=self.solver,
        )
        self.status = status
        if status == "optimal":
            self.template_b = support
            self.initialized = True
        return status

    def update(self, phi: Sequence[float], delta: float) -> str:
        """Recursively update support values with one new measurement."""
        if not self.initialized:
            raise RuntimeError("RecursiveTemplateSME must be initialized before update")
        support, status = update_template_support(
            self.template_A,
            self.template_b,
            phi,
            delta,
            self.w_max,
            solver=self.solver,
        )
        self.status = status
        if status == "optimal":
            self.template_b = support
        return status

    def coordinate_bounds(self) -> tuple[np.ndarray, np.ndarray, str]:
        return coordinate_bounds_from_template(
            self.template_A,
            self.template_b,
            solver=self.solver,
        )

    def coordinate_width(self) -> float:
        return coordinate_width_from_template(
            self.template_A,
            self.template_b,
            solver=self.solver,
        )


@dataclass
class SMECurveSummary:
    """Aggregated SME width curves and first-hit statistics."""

    width_norm_seqs: List[np.ndarray]
    reach_T: List[Optional[int]]
    reach_metric: List[float]
    mean_width_norm: np.ndarray
    std_width_norm: np.ndarray
    wall_s: float
    wall_s_by_traj: List[float]
    reach_T_mean: float


def final_finite_value(seq: Sequence[float]) -> float:
    """Last finite value in a curve, or nan if the curve has no finite entries."""
    values = np.asarray(seq, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite[-1]) if finite.size else float("nan")


def summarize_sme_stack(
    width_norm_seqs: Sequence[np.ndarray],
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise nan mean/std for SME width curves."""
    if not width_norm_seqs:
        nan_curve = np.full(int(horizon), np.nan, dtype=float)
        return nan_curve, nan_curve.copy()

    stack = np.stack(width_norm_seqs, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)


def exact_sme_coordinate_width_sequence(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
    *,
    min_T: int = 2,
    theta_star: Optional[np.ndarray] = None,
    tol: Optional[float] = None,
    max_T: Optional[int] = None,
    update_every: int = 1,
    stop_at_tol: bool = False,
    solver: str = "linprog",
) -> tuple[np.ndarray, Optional[int], float]:
    """Normalized exact SME coordinate-width curve."""
    phi = np.asarray(phi_s_u, dtype=float)
    n = min(len(delta_s), phi.shape[0])
    if max_T is not None:
        n = min(n, int(max_T))

    out = np.full(n, np.nan, dtype=float)
    theta_norm = (
        float(np.linalg.norm(np.asarray(theta_star, dtype=float)) + 1e-12)
        if theta_star is not None
        else 1.0
    )
    step = max(1, int(update_every))
    reach_T: Optional[int] = None
    reach_metric = float("nan")

    for T in range(int(min_T), n + 1):
        if (T - int(min_T)) % step != 0 and T != n:
            continue
        width = sme_coordinate_width_at_T(
            delta_s,
            phi,
            T,
            w_max,
            feasible_fallback=theta_star,
            verbose=False,
            solver=solver,
        )
        metric = width / theta_norm
        out[T - 1] = metric
        if reach_T is None and tol is not None and np.isfinite(metric) and metric <= tol:
            reach_T = T
            reach_metric = metric
            if stop_at_tol:
                break

    return out, reach_T, reach_metric


def _monotone_sampled_sme_coordinate_width_sequence(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
    *,
    sample_size: int,
    mode: str,
    min_T: int = 2,
    theta_star: Optional[np.ndarray] = None,
    tol: Optional[float] = None,
    max_T: Optional[int] = None,
    update_every: int = 1,
    stop_at_tol: bool = False,
    seed: Optional[int] = 0,
    solver: str = "linprog",
) -> tuple[np.ndarray, Optional[int], float]:
    """Exact SME on sampled measurements, accepting only shrinking widths."""
    phi = np.asarray(phi_s_u, dtype=float)
    delta = np.asarray(delta_s, dtype=float)
    n = min(delta.shape[0], phi.shape[0])
    if max_T is not None:
        n = min(n, int(max_T))

    out = np.full(n, np.nan, dtype=float)
    theta_norm = (
        float(np.linalg.norm(np.asarray(theta_star, dtype=float)) + 1e-12)
        if theta_star is not None
        else 1.0
    )
    rng = np.random.default_rng(seed)
    sample = max(1, int(sample_size))
    step = max(1, int(update_every))
    reach_T: Optional[int] = None
    reach_metric = float("nan")
    best_metric = float("inf")

    mode_normalized = str(mode).lower()
    if mode_normalized not in {"rollout", "random"}:
        raise ValueError('mode must be "rollout" or "random"')

    for T in range(int(min_T), n + 1):
        if (T - int(min_T)) % step != 0 and T != n:
            if np.isfinite(best_metric):
                out[T - 1] = best_metric
            continue

        if mode_normalized == "rollout":
            selected_start = max(0, T - sample)
            selected_idx = np.arange(selected_start, T, dtype=int)
        else:
            n_sample = min(sample, T)
            selected_idx = np.sort(rng.choice(T, size=n_sample, replace=False))

        selected_delta = delta[selected_idx]
        selected_phi = phi[selected_idx]
        width = sme_coordinate_width_at_T(
            selected_delta,
            selected_phi,
            len(selected_delta),
            w_max,
            feasible_fallback=theta_star,
            verbose=False,
            solver=solver,
        )
        metric = width / theta_norm
        if np.isfinite(metric) and metric < best_metric:
            best_metric = metric
        if np.isfinite(best_metric):
            out[T - 1] = best_metric
        if reach_T is None and tol is not None and np.isfinite(metric) and metric <= tol:
            reach_T = T
            reach_metric = best_metric
            if stop_at_tol:
                break

    return out, reach_T, reach_metric


def roll_out_sme_coordinate_width_sequence(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
    *,
    recent_window: int = 100,
    min_T: int = 2,
    theta_star: Optional[np.ndarray] = None,
    tol: Optional[float] = None,
    max_T: Optional[int] = None,
    update_every: int = 1,
    stop_at_tol: bool = False,
    solver: str = "linprog",
) -> tuple[np.ndarray, Optional[int], float]:
    """Roll-out SME on the latest fixed window, kept monotone by width."""
    return _monotone_sampled_sme_coordinate_width_sequence(
        delta_s,
        phi_s_u,
        w_max,
        sample_size=recent_window,
        mode="rollout",
        min_T=min_T,
        theta_star=theta_star,
        tol=tol,
        max_T=max_T,
        update_every=update_every,
        stop_at_tol=stop_at_tol,
        seed=None,
        solver=solver,
    )


def random_sme_coordinate_width_sequence(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
    *,
    sample_size: int = 100,
    min_T: int = 2,
    theta_star: Optional[np.ndarray] = None,
    tol: Optional[float] = None,
    max_T: Optional[int] = None,
    update_every: int = 1,
    stop_at_tol: bool = False,
    seed: Optional[int] = 0,
    solver: str = "linprog",
) -> tuple[np.ndarray, Optional[int], float]:
    """Random SME using a fixed-size subset from the past trajectory."""
    return _monotone_sampled_sme_coordinate_width_sequence(
        delta_s,
        phi_s_u,
        w_max,
        sample_size=sample_size,
        mode="random",
        min_T=min_T,
        theta_star=theta_star,
        tol=tol,
        max_T=max_T,
        update_every=update_every,
        stop_at_tol=stop_at_tol,
        seed=seed,
        solver=solver,
    )


def fixed_sample_exact_sme_coordinate_width_sequence(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
    *,
    recent_window: int = 100,
    history_sample: int = 400,
    min_T: int = 2,
    theta_star: Optional[np.ndarray] = None,
    tol: Optional[float] = None,
    max_T: Optional[int] = None,
    update_every: int = 1,
    stop_at_tol: bool = False,
    seed: Optional[int] = 0,
    solver: str = "linprog",
) -> tuple[np.ndarray, Optional[int], float]:
    """Backward-compatible wrapper for the monotone roll-out SME curve."""
    _ = history_sample, seed
    return roll_out_sme_coordinate_width_sequence(
        delta_s,
        phi_s_u,
        w_max,
        recent_window=recent_window,
        min_T=min_T,
        theta_star=theta_star,
        tol=tol,
        max_T=max_T,
        update_every=update_every,
        stop_at_tol=stop_at_tol,
        solver=solver,
    )


def approximate_sme_coordinate_width_sequence(
    delta_s: Sequence[float],
    phi_s_u: np.ndarray,
    w_max: float,
    *,
    n_sides: int = 4,
    min_T: int = 2,
    theta_star: Optional[np.ndarray] = None,
    tol: Optional[float] = None,
    max_T: Optional[int] = None,
    stop_at_tol: bool = False,
    solver: str = "linprog",
) -> tuple[np.ndarray, Optional[int], float, RecursiveTemplateSME]:
    """Return the normalized recursive template-SME width curve.

    If ``theta_star`` is omitted, the returned curve is unnormalized. ``reach_T``
    is the first 1-based sample index whose metric is at or below ``tol``.
    """
    phi = np.asarray(phi_s_u, dtype=float)
    delta = np.asarray(delta_s, dtype=float)
    n = min(delta.shape[0], phi.shape[0])
    if max_T is not None:
        n = min(n, int(max_T))

    out = np.full(n, np.nan, dtype=float)
    estimator = RecursiveTemplateSME.regular(
        n_sides=n_sides,
        w_max=w_max,
        solver=solver,
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

        if estimator.status != "optimal":
            continue

        metric = estimator.coordinate_width() / theta_norm
        out[T - 1] = metric
        if reach_T is None and tol is not None and np.isfinite(metric) and metric <= tol:
            reach_T = T
            reach_metric = metric
            if stop_at_tol:
                break

    return out, reach_T, reach_metric, estimator


def _run_sme_comparison(
    curve_fn: Callable[..., tuple],
    delta_s_list: Sequence[Sequence[float]],
    phi_s_u_list: Sequence[np.ndarray],
    *,
    label: str,
    n_epoch_sme: int,
    horizon: int,
    print_horizon_label: str,
    **curve_kwargs,
) -> SMECurveSummary:
    """Run an SME curve function over trajectories and print concise progress."""
    t_start = time.perf_counter()
    width_norm_seqs: List[np.ndarray] = []
    reach_T: List[Optional[int]] = []
    reach_metric: List[float] = []
    wall_s_by_traj: List[float] = []

    print(f"{label} until {print_horizon_label}...")
    for e in range(int(n_epoch_sme)):
        print(f"  trajectory {e + 1} / {n_epoch_sme}", end="")
        t_traj = time.perf_counter()
        result = curve_fn(delta_s_list[e], phi_s_u_list[e], **curve_kwargs)
        wall_s_by_traj.append(time.perf_counter() - t_traj)
        width_norm_e = result[0]
        reach_T_e = result[1]
        reach_metric_e = result[2]
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

    return SMECurveSummary(
        width_norm_seqs=width_norm_seqs,
        reach_T=reach_T,
        reach_metric=reach_metric,
        mean_width_norm=mean_width_norm,
        std_width_norm=std_width_norm,
        wall_s=wall_s,
        wall_s_by_traj=wall_s_by_traj,
        reach_T_mean=reach_T_mean,
    )


def run_exact_sme_curves(
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
    update_every: int = 1,
    stop_at_tol: bool = False,
    solver: str = "linprog",
) -> SMECurveSummary:
    """Run exact SME curves over multiple trajectories."""
    return _run_sme_comparison(
        exact_sme_coordinate_width_sequence,
        delta_s_list,
        phi_s_u_list,
        label="Exact SME",
        n_epoch_sme=n_epoch_sme,
        horizon=horizon,
        print_horizon_label=print_horizon_label,
        w_max=w_max,
        min_T=min_T,
        theta_star=theta_star,
        tol=tol,
        max_T=horizon,
        update_every=update_every,
        stop_at_tol=stop_at_tol,
        solver=solver,
    )


def run_template_sme_curves(
    delta_s_list: Sequence[Sequence[float]],
    phi_s_u_list: Sequence[np.ndarray],
    *,
    n_epoch_sme: int,
    horizon: int,
    print_horizon_label: str,
    w_max: float,
    n_sides: int,
    min_T: int,
    theta_star: np.ndarray,
    tol: Optional[float],
    stop_at_tol: bool = False,
    solver: str = "linprog",
) -> SMECurveSummary:
    """Run recursive template-SME curves over multiple trajectories."""
    return _run_sme_comparison(
        approximate_sme_coordinate_width_sequence,
        delta_s_list,
        phi_s_u_list,
        label=f"Template SME (m={n_sides})",
        n_epoch_sme=n_epoch_sme,
        horizon=horizon,
        print_horizon_label=print_horizon_label,
        w_max=w_max,
        n_sides=n_sides,
        min_T=min_T,
        theta_star=theta_star,
        tol=tol,
        max_T=horizon,
        stop_at_tol=stop_at_tol,
        solver=solver,
    )


def run_roll_out_sme_curves(
    delta_s_list: Sequence[Sequence[float]],
    phi_s_u_list: Sequence[np.ndarray],
    *,
    n_epoch_sme: int,
    horizon: int,
    print_horizon_label: str,
    w_max: float,
    recent_window: int,
    min_T: int,
    theta_star: np.ndarray,
    tol: Optional[float],
    update_every: int = 1,
    stop_at_tol: bool = False,
    solver: str = "linprog",
) -> SMECurveSummary:
    """Run monotone roll-out SME curves over multiple trajectories."""
    return _run_sme_comparison(
        roll_out_sme_coordinate_width_sequence,
        delta_s_list,
        phi_s_u_list,
        label=f"Roll out SME (H={recent_window})",
        n_epoch_sme=n_epoch_sme,
        horizon=horizon,
        print_horizon_label=print_horizon_label,
        w_max=w_max,
        recent_window=recent_window,
        min_T=min_T,
        theta_star=theta_star,
        tol=tol,
        max_T=horizon,
        update_every=update_every,
        stop_at_tol=stop_at_tol,
        solver=solver,
    )


def run_random_sme_curves(
    delta_s_list: Sequence[Sequence[float]],
    phi_s_u_list: Sequence[np.ndarray],
    *,
    n_epoch_sme: int,
    horizon: int,
    print_horizon_label: str,
    w_max: float,
    sample_size: int,
    min_T: int,
    theta_star: np.ndarray,
    tol: Optional[float],
    update_every: int = 1,
    stop_at_tol: bool = False,
    seed: int = 0,
    solver: str = "linprog",
) -> SMECurveSummary:
    """Run monotone random SME curves over multiple trajectories."""
    t_start = time.perf_counter()
    width_norm_seqs: List[np.ndarray] = []
    reach_T: List[Optional[int]] = []
    reach_metric: List[float] = []
    wall_s_by_traj: List[float] = []
    label = f"Random SME (N={sample_size})"

    print(f"{label} until {print_horizon_label}...")
    for e in range(int(n_epoch_sme)):
        print(f"  trajectory {e + 1} / {n_epoch_sme}", end="")
        t_traj = time.perf_counter()
        width_norm_e, reach_T_e, reach_metric_e = random_sme_coordinate_width_sequence(
            delta_s_list[e],
            phi_s_u_list[e],
            w_max,
            sample_size=sample_size,
            min_T=min_T,
            theta_star=theta_star,
            tol=tol,
            max_T=horizon,
            update_every=update_every,
            stop_at_tol=stop_at_tol,
            seed=int(seed) + e,
            solver=solver,
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

    return SMECurveSummary(
        width_norm_seqs=width_norm_seqs,
        reach_T=reach_T,
        reach_metric=reach_metric,
        mean_width_norm=mean_width_norm,
        std_width_norm=std_width_norm,
        wall_s=wall_s,
        wall_s_by_traj=wall_s_by_traj,
        reach_T_mean=reach_T_mean,
    )


def run_fixed_sample_exact_sme_curves(
    delta_s_list: Sequence[Sequence[float]],
    phi_s_u_list: Sequence[np.ndarray],
    *,
    n_epoch_sme: int,
    horizon: int,
    print_horizon_label: str,
    w_max: float,
    recent_window: int,
    history_sample: int,
    min_T: int,
    theta_star: np.ndarray,
    tol: Optional[float],
    update_every: int = 1,
    stop_at_tol: bool = False,
    seed: int = 0,
    solver: str = "linprog",
) -> SMECurveSummary:
    """Backward-compatible wrapper for monotone roll-out SME curves."""
    _ = history_sample, seed
    return run_roll_out_sme_curves(
        delta_s_list,
        phi_s_u_list,
        n_epoch_sme=n_epoch_sme,
        horizon=horizon,
        print_horizon_label=print_horizon_label,
        w_max=w_max,
        recent_window=recent_window,
        min_T=min_T,
        theta_star=theta_star,
        tol=tol,
        update_every=update_every,
        stop_at_tol=stop_at_tol,
        solver=solver,
    )
