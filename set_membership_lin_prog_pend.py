"""Set-membership estimation (SME) for pendulum sys-id via coordinate LP bounds."""

from __future__ import annotations

from functools import lru_cache
from typing import List, Optional, Sequence

import numpy as np
from scipy.optimize import linprog

# Legacy defaults (pendulum_SME.ipynb)
m = 0.1
l = 0.5
ground_truth = [1 / l, 1 / m / l / l]

SUPPORTED_LP_SOLVERS = {"linprog", "gurobi"}


@lru_cache(maxsize=1)
def _gurobi_env():
    import gurobipy as gp

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)
    env.start()
    return env


def _normalize_solver(solver: str) -> str:
    solver_name = str(solver).lower()
    if solver_name == "highs":
        solver_name = "linprog"
    if solver_name not in SUPPORTED_LP_SOLVERS:
        raise ValueError(
            f"solver must be one of {sorted(SUPPORTED_LP_SOLVERS)}, got {solver!r}"
        )
    return solver_name


def solve_sme_lp(
    c: Sequence[float],
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    *,
    bounds: Optional[Sequence[tuple[Optional[float], Optional[float]]]] = None,
    solver: str = "linprog",
) -> tuple[bool, np.ndarray, str]:
    """Solve a small SME LP using SciPy/HiGHS or Gurobi."""
    solver_name = _normalize_solver(solver)
    c_arr = np.asarray(c, dtype=float)
    A_arr = np.asarray(A_ub, dtype=float)
    b_arr = np.asarray(b_ub, dtype=float)

    if bounds is None:
        bounds = [(None, None)] * c_arr.shape[0]

    if solver_name == "linprog":
        sol = linprog(
            c_arr,
            A_ub=A_arr,
            b_ub=b_arr,
            bounds=bounds,
            method="highs",
        )
        x = np.asarray(sol.x, dtype=float) if sol.x is not None else np.full_like(c_arr, np.nan)
        return bool(sol.success), x, str(sol.message)

    try:
        import gurobipy as gp
    except ImportError as exc:
        raise ImportError(
            "SME solver='gurobi' requires gurobipy to be installed."
        ) from exc

    model = gp.Model(env=_gurobi_env())
    model.Params.OutputFlag = 0
    lower = [
        -gp.GRB.INFINITY if lo is None else float(lo)
        for lo, _hi in bounds
    ]
    upper = [
        gp.GRB.INFINITY if hi is None else float(hi)
        for _lo, hi in bounds
    ]
    x_var = model.addMVar(shape=c_arr.shape[0], lb=lower, ub=upper, obj=c_arr)
    if A_arr.shape[0] > 0:
        model.addMConstr(A_arr, x_var, "<", b_arr)
    model.ModelSense = gp.GRB.MINIMIZE
    model.optimize()

    if model.Status == gp.GRB.OPTIMAL:
        return True, np.asarray(x_var.X, dtype=float), "optimal"
    status_name = {
        gp.GRB.INFEASIBLE: "infeasible",
        gp.GRB.UNBOUNDED: "unbounded",
        gp.GRB.INF_OR_UNBD: "infeasible or unbounded",
    }.get(model.Status, f"gurobi status {model.Status}")
    return False, np.full_like(c_arr, np.nan), status_name


def _sme_ub_constraints_array(
    Delta_S: Sequence[float],
    Phi_S_U: np.ndarray,
    w_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized G theta <= h constraints for LP-only SME size calculations."""
    phi = np.asarray(Phi_S_U, dtype=float)
    delta = np.asarray(Delta_S, dtype=float)
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


def uncertainty_set_coordinate_bounds(
    Delta_S: Sequence[float],
    Phi_S_U: np.ndarray,
    w_max: float,
    *,
    solver: str = "linprog",
) -> tuple[np.ndarray, np.ndarray, str]:
    """Coordinate-wise min/max bounds for the SME feasible set via 4 LPs."""
    A_ub, b_ub = _sme_ub_constraints_array(Delta_S, Phi_S_U, w_max)
    if A_ub.shape[0] == 0:
        nan_bounds = np.full(2, np.nan, dtype=float)
        return nan_bounds, nan_bounds, "empty"

    lower = np.full(2, np.nan, dtype=float)
    upper = np.full(2, np.nan, dtype=float)
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

        c_max = np.zeros(2, dtype=float)
        c_max[dim] = -1.0
        success_max, x_max, message_max = solve_sme_lp(
            c_max,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=variable_bounds,
            solver=solver,
        )
        if not success_max:
            return lower, upper, message_max
        upper[dim] = float(x_max[dim])

    return lower, upper, "optimal"


def uncertainty_set_coordinate_width(
    Delta_S: Sequence[float],
    Phi_S_U: np.ndarray,
    w_max: float,
    *,
    solver: str = "linprog",
) -> float:
    """Largest coordinate width, max_j(theta_j^max - theta_j^min), via 4 LPs."""
    lower, upper, status = uncertainty_set_coordinate_bounds(
        Delta_S,
        Phi_S_U,
        w_max,
        solver=solver,
    )
    if status != "optimal":
        return float("nan")
    return float(np.max(upper - lower))


def sme_coordinate_width_at_T(
    Delta_S: Sequence[float],
    Phi_S_U: np.ndarray,
    T: int,
    w_max: float,
    *,
    feasible_fallback: Optional[np.ndarray] = None,
    verbose: bool = False,
    solver: str = "linprog",
) -> float:
    """Passive SME size using 4 LPs over coordinate min/max bounds."""
    _ = feasible_fallback, verbose  # Kept for API compatibility with old callers.
    return uncertainty_set_coordinate_width(
        Delta_S[:T],
        np.asarray(Phi_S_U[:T], dtype=float),
        w_max,
        solver=solver,
    )


def sme_coordinate_width_sequence(
    Delta_S: Sequence[float],
    Phi_S_U: np.ndarray,
    w_max: float,
    *,
    min_T: int = 2,
    feasible_fallback: Optional[np.ndarray] = None,
    verbose: bool = False,
    progress_every: Optional[int] = 5000,
    update_every: Optional[int] = None,
    solver: str = "linprog",
) -> np.ndarray:
    """
    LP coordinate-width size after each sample index (1-based T).

    Index t-1 holds the size using first t samples; indices < min_T-1 are nan.

    If ``update_every`` is set, the feasible set is recomputed only every that many
    samples; between updates the last size is held constant.
    """
    if update_every is not None and update_every > 1:
        return sme_coordinate_width_sequence_episode(
            Delta_S,
            Phi_S_U,
            w_max,
            min_T=min_T,
            feasible_fallback=feasible_fallback,
            verbose=verbose,
            progress_every=progress_every,
            update_every=update_every,
            solver=solver,
        )

    phi = np.asarray(Phi_S_U, dtype=float)
    n = min(len(Delta_S), phi.shape[0])
    out = np.full(n, np.nan, dtype=float)

    for T in range(min_T, n + 1):
        out[T - 1] = sme_coordinate_width_at_T(
            Delta_S,
            phi,
            T,
            w_max,
            feasible_fallback=feasible_fallback,
            verbose=verbose,
            solver=solver,
        )
        if progress_every and T % progress_every == 0:
            print("  SME T =", T)

    return out


def sme_coordinate_width_sequence_episode(
    Delta_S: Sequence[float],
    Phi_S_U: np.ndarray,
    w_max: float,
    *,
    min_T: int = 2,
    feasible_fallback: Optional[np.ndarray] = None,
    verbose: bool = False,
    progress_every: Optional[int] = 5000,
    update_every: int,
    solver: str = "linprog",
) -> np.ndarray:
    """SME LP coordinate-width curve with batch updates every ``update_every`` samples."""
    phi = np.asarray(Phi_S_U, dtype=float)
    n = min(len(Delta_S), phi.shape[0])
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out

    step = max(1, int(update_every))
    last_width = np.nan
    batch_start = 0

    for T in range(1, n + 1):
        if T < min_T:
            continue
        if (T - batch_start) >= step or T == n:
            last_width = sme_coordinate_width_at_T(
                Delta_S,
                phi,
                T,
                w_max,
                feasible_fallback=feasible_fallback,
                verbose=verbose,
                solver=solver,
            )
            batch_start = T
            if progress_every and T % progress_every == 0:
                print("  SME T =", T)
        out[T - 1] = last_width

    return out


def mean_std_width_curves(
    width_seqs: List[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Element-wise nan-mean and nan-std across trajectories."""
    stack = np.stack(width_seqs, axis=0)
    return np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)


def normalize_width_curve(
    width: np.ndarray,
    theta_star: np.ndarray,
) -> np.ndarray:
    """Scale a passive SME size curve by ||theta*||_2."""
    norm = float(np.linalg.norm(np.asarray(theta_star, dtype=float)) + 1e-12)
    return np.asarray(width, dtype=float) / norm
