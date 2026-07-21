"""
Active data collection for pendulum system identification (Mania et al.,
arXiv:2006.10277, Appendix D Algorithm 2).

    (omega_{t+1} - omega_t) / dt = Theta^* phi_t + w_t
    Theta^* = [1/l, 1/(m l^2)],   phi_t = [-g sin(alpha_t), u_t]^T.

State x = [omega, alpha] (paper layout).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from lse_pend import error_theta_l2_rel
from pendulum_dynamics import dt, g, generate_w
from pendulum_RLS import RecursiveLS

# ---------------------------------------------------------------------------
# Features: phi_t and phi_dt = dt * phi_t (for lse_pend.run_lst compatibility)
# ---------------------------------------------------------------------------


def phi_reg(x: np.ndarray, u: float) -> np.ndarray:
    """phi_t = [-g sin(alpha_t), u_t]."""
    return np.array([-g * math.sin(float(x[1])), float(u)], dtype=float)


def phi_dt(x: np.ndarray, u: float) -> np.ndarray:
    """phi_dt = dt * phi_t."""
    return dt * phi_reg(x, u)


def fit_theta_lse(
    phi_rows: np.ndarray,
    y_delta_omega: np.ndarray,
    ridge: float = 1e-8,
) -> np.ndarray:
    """LSE: y_rate = Theta phi,  y_rate = Delta_omega / dt;  Theta^T = pinv(Phi) y."""
    Phi = np.asarray(phi_rows, dtype=float)
    y_rate = np.asarray(y_delta_omega, dtype=float).ravel() / dt
    if Phi.shape[0] == 0:
        return np.zeros(2, dtype=float)
    if ridge > 0:
        gram = Phi.T @ Phi + ridge * np.eye(2)
        return np.linalg.solve(gram, Phi.T @ y_rate)
    return np.linalg.pinv(Phi) @ y_rate


def theta_apply(phi: np.ndarray, theta_hat: np.ndarray) -> float:
    return float(np.asarray(theta_hat, dtype=float).ravel() @ np.asarray(phi, dtype=float).ravel())


def tracking_residual(
    theta_hat: np.ndarray, phi: np.ndarray, phi_ref: np.ndarray
) -> float:
    """|Theta (phi - phi_ref)| — Algorithm 2 tracking objective."""
    return abs(theta_apply(phi - phi_ref, theta_hat))


def true_step(
    x: np.ndarray,
    u: float,
    m: float,
    l: float,
    w1: float,
    w2: float,
) -> np.ndarray:
    omega, alpha = float(x[0]), float(x[1])
    alpha_dot = omega + w1
    omega_dot = -g * math.sin(alpha) / l + u / (m * l * l) + w2
    return np.array([omega + dt * omega_dot, alpha + dt * alpha_dot], dtype=float)


def identified_step(x: np.ndarray, u: float, theta_hat: np.ndarray) -> np.ndarray:
    d_omega = dt * theta_apply(phi_reg(x, u), theta_hat)
    return np.array([x[0] + d_omega, x[1] + dt * x[0]], dtype=float)


def rollout_ref_states_planning(
    x0: np.ndarray,
    u_pre: List[float],
    theta_hat: np.ndarray,
) -> List[np.ndarray]:
    xs: List[np.ndarray] = [x0.copy()]
    x = x0.copy()
    for uj in u_pre:
        x = identified_step(x, uj, theta_hat)
        xs.append(x.copy())
    return xs


def min_eigenvector_symmetric(M: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    Mreg = 0.5 * (M + M.T) + ridge * np.eye(M.shape[0])
    w, V = np.linalg.eigh(Mreg)
    return V[:, 0] / np.linalg.norm(V[:, 0])


def _leverage(phi_v: np.ndarray, gram_inv: np.ndarray) -> float:
    return float(phi_v @ gram_inv @ phi_v)


def _terminal_ok(
    phi_term: np.ndarray,
    v: np.ndarray,
    gram_inv: np.ndarray,
    alpha: float,
    beta: float,
) -> bool:
    return abs(float(phi_term @ v)) >= alpha / 2.0 or _leverage(phi_term, gram_inv) >= beta


def argmax_leverage(
    x: np.ndarray,
    gram_inv: np.ndarray,
    bu: float,
) -> Tuple[float, np.ndarray]:
    def neg_leverage(u: float) -> float:
        return -_leverage(phi_reg(x, u), gram_inv)

    res = minimize_scalar(neg_leverage, bounds=(-bu, bu), method="bounded")
    u_best = float(res.x)
    return u_best, phi_reg(x, u_best)


def argmin_tracking(
    x: np.ndarray,
    phi_ref: np.ndarray,
    theta_hat: np.ndarray,
    bu: float,
) -> float:
    def tracking_obj(u: float) -> float:
        return tracking_residual(theta_hat, phi_reg(x, u), phi_ref)

    res = minimize_scalar(tracking_obj, bounds=(-bu, bu), method="bounded")
    return float(res.x)


def random_trajectory_plan(
    x0: np.ndarray,
    theta_hat: np.ndarray,
    gram_inv: np.ndarray,
    v: np.ndarray,
    alpha: float,
    beta: float,
    H: int,
    n_random_trials: int,
    bu: float,
    rng: np.random.Generator,
) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """Random-search planning oracle (Algorithm 2)."""
    for _ in range(n_random_trials):
        r = int(rng.integers(0, H + 1))
        x = x0.copy()
        u_pre: List[float] = []
        for _j in range(r):
            uj = float(rng.uniform(-bu, bu))
            u_pre.append(uj)
            x = identified_step(x, uj, theta_hat)
        ur = float(rng.uniform(-bu, bu))
        phi_term = phi_reg(x, ur)
        if _terminal_ok(phi_term, v, gram_inv, alpha, beta):
            return True, {
                "r": r,
                "u_pre": u_pre,
                "u_term": ur,
                "x_ref": rollout_ref_states_planning(x0, u_pre, theta_hat),
                "u_ref": u_pre + [ur],
            }
    return False, None


@dataclass
class ActiveExploreResult:
    theta_hat: np.ndarray
    phi_rows: np.ndarray
    psi_rows: np.ndarray
    y_delta_omega: np.ndarray
    states: List[np.ndarray]
    planning_failed: bool
    planning_fail_message: str = ""
    steps: int = 0
    n_segments: int = 0
    n_rls_updates: int = 0
    avg_steps_per_update: float = 0.0
    collection_wall_s: float = 0.0
    reached_tol: bool = False
    samples_at_tol: int = 0
    wall_s_at_tol: float = 0.0


def run_active_learning(
    x0_paper: np.ndarray,
    m: float,
    l: float,
    T_budget: int,
    *,
    H: int = 15,
    alpha: float = 0.05,
    beta: float = 0.25,
    bu: float = 1.0,
    n_plan_samples: int = 2000,
    warmup: int = 20,
    ridge: float = 1e-6,
    distr: str = "trunc_guass",
    param_w: Tuple[float, float, float] = (0.0, 0.1, 10.0),
    mult_w: Tuple[float, float] = (1.0, 1.0),
    seed_w: int = 0,
    theta_star: Optional[np.ndarray] = None,
    theta_tol: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> ActiveExploreResult:
    """Algorithm 2 with random-search planning; RLS updated once per episode."""
    if rng is None:
        rng = np.random.default_rng()

    t_collect_start = time.perf_counter()

    w1_all, w2_all = generate_w(
        distr,
        max(T_budget + 10, warmup + 10),
        seed_w,
        param_w[0],
        param_w[1],
        param_w[2],
        -1.0,
        1.0,
    )
    mw1, mw2 = mult_w

    phi_list: List[np.ndarray] = []
    phidt_list: List[np.ndarray] = []
    y_list: List[float] = []
    states: List[np.ndarray] = [x0_paper.copy()]

    x = x0_paper.copy()
    t_global = 0
    n_segments = 0
    n_rls_updates = 0
    planning_failed = False
    fail_msg = ""
    reached_tol = False
    samples_at_tol = 0
    wall_s_at_tol = 0.0

    rls = RecursiveLS(n_features=2, ridge=ridge)
    theta_hat = np.zeros(2, dtype=float)
    episode_phidt: List[np.ndarray] = []
    episode_y: List[float] = []

    def append_transition(x_cur: np.ndarray, u: float, widx: int) -> None:
        nonlocal t_global
        w1 = mw1 * w1_all[widx]
        w2 = mw2 * w2_all[widx]
        x_next = true_step(x_cur, u, m, l, w1, w2)
        phi = phi_reg(x_cur, u)
        phi_d = phi_dt(x_cur, u)
        y = float(x_next[0] - x_cur[0])
        phi_list.append(phi)
        phidt_list.append(phi_d)
        y_list.append(y)
        episode_phidt.append(phi_d)
        episode_y.append(y)
        states.append(x_next.copy())
        x_cur[:] = x_next
        t_global += 1

    def tol_reached() -> bool:
        return (
            theta_tol is not None
            and theta_star is not None
            and rls.n_samples > 0
            and error_theta_l2_rel(theta_hat, theta_star) <= theta_tol
        )

    def flush_episode_to_rls() -> bool:
        """Apply buffered episode samples to RLS; return True if precision tol met."""
        nonlocal theta_hat, n_rls_updates, reached_tol, samples_at_tol, wall_s_at_tol
        if not episode_phidt:
            return False
        for phi_d, y in zip(episode_phidt, episode_y):
            rls.update(phi_d, y)
        episode_phidt.clear()
        episode_y.clear()
        n_rls_updates += 1
        if rls.n_samples > 0:
            theta_hat = rls.theta_hat()
        if tol_reached():
            reached_tol = True
            samples_at_tol = t_global
            wall_s_at_tol = time.perf_counter() - t_collect_start
            return True
        return False

    for _ in range(warmup):
        if t_global >= T_budget:
            break
        append_transition(x, float(rng.uniform(-bu, bu)), t_global)
    if flush_episode_to_rls():
        collection_wall_s = time.perf_counter() - t_collect_start
        avg_steps = float(t_global / n_rls_updates) if n_rls_updates > 0 else 0.0
        return ActiveExploreResult(
            theta_hat=theta_hat,
            phi_rows=np.vstack(phi_list) if phi_list else np.zeros((0, 2)),
            psi_rows=np.vstack(phidt_list) if phidt_list else np.zeros((0, 2)),
            y_delta_omega=np.array(y_list, dtype=float),
            states=states,
            planning_failed=False,
            steps=t_global,
            n_segments=n_segments,
            n_rls_updates=n_rls_updates,
            avg_steps_per_update=avg_steps,
            collection_wall_s=collection_wall_s,
            reached_tol=True,
            samples_at_tol=samples_at_tol,
            wall_s_at_tol=wall_s_at_tol,
        )

    while t_global < T_budget and not planning_failed:
        gram = rls.gram + ridge * np.eye(2)
        try:
            gram_inv = np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            gram_inv = np.linalg.pinv(gram)
        v = min_eigenvector_symmetric(gram)

        ok, plan = random_trajectory_plan(
            x.copy(),
            theta_hat,
            gram_inv,
            v,
            alpha,
            beta,
            H,
            n_plan_samples,
            bu,
            rng,
        )
        if not ok:
            planning_failed = True
            fail_msg = (
                f"Active planning failed (n_random={n_plan_samples}, H={H}). "
                "Try larger n_plan_samples, H, bu, or relax alpha/beta."
            )
            break

        assert plan is not None
        r = plan["r"]

        # # debug
        # print(f"r: {r}")

        x_refs = plan["x_ref"]
        u_refs = plan["u_ref"]
        n_segments += 1

        j = 0
        while j <= r and t_global < T_budget:
            ug, _ = argmax_leverage(x, gram_inv, bu)
            if _leverage(phi_reg(x, ug), gram_inv) >= beta:
                append_transition(x, ug, t_global)
                break
            if j <= r - 1:
                u_track = argmin_tracking(
                    x, phi_reg(x_refs[j], u_refs[j]), theta_hat, bu
                )
                append_transition(x, u_track, t_global)
                j += 1
            else:
                append_transition(x, u_refs[r], t_global)
                break

        if flush_episode_to_rls():
            break

    collection_wall_s = time.perf_counter() - t_collect_start
    avg_steps = float(t_global / n_rls_updates) if n_rls_updates > 0 else 0.0

    return ActiveExploreResult(
        theta_hat=theta_hat,
        phi_rows=np.vstack(phi_list) if phi_list else np.zeros((0, 2)),
        psi_rows=np.vstack(phidt_list) if phidt_list else np.zeros((0, 2)),
        y_delta_omega=np.array(y_list, dtype=float),
        states=states,
        planning_failed=planning_failed,
        planning_fail_message=fail_msg,
        steps=t_global,
        n_segments=n_segments,
        n_rls_updates=n_rls_updates,
        avg_steps_per_update=avg_steps,
        collection_wall_s=collection_wall_s,
        reached_tol=reached_tol,
        samples_at_tol=samples_at_tol,
        wall_s_at_tol=wall_s_at_tol,
    )


def update_every_from_active_results(results: Sequence[ActiveExploreResult]) -> int:
    """Mean steps per RLS update over successful active runs (passive RLS/SME cadence)."""
    avgs = [
        r.avg_steps_per_update
        for r in results
        if not r.planning_failed and r.n_rls_updates > 0
    ]
    return max(1, int(round(float(np.mean(avgs))))) if avgs else 1
