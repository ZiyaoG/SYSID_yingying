"""Scalar dynamics with the non-real-analytic bump feature f(x)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.stats import truncnorm

from pendulum_dynamics import generate_u


def bump_function(x: float | np.ndarray) -> np.ndarray:
    """f(x) = exp(-1/x^2) for x > 0, else 0 (C^infty but not real-analytic at 0)."""
    x_arr = np.asarray(x, dtype=float)
    out = np.zeros_like(x_arr, dtype=float)
    pos = x_arr > 0.0
    out[pos] = np.exp(-1.0 / x_arr[pos] ** 2)
    return out


def theta_star_values(a_star: float, b_star: float) -> np.ndarray:
    """Ground-truth regression parameters [a, b]."""
    return np.array([a_star, b_star], dtype=float)


def phi_features(x: float, u: float) -> np.ndarray:
    """Feature vector phi(x, u) = [f(x), f(u)]."""
    return np.array([float(bump_function(x)), float(bump_function(u))], dtype=float)


def generate_w(
    distr: str,
    time_hor: int,
    seed: int,
    mean: float,
    std: float,
    w_max: float,
    lb: float,
    ub: float,
) -> np.ndarray:
    """Disturbance w_t (matches pendulum ``generate_w`` style)."""
    if distr == "trunc_guass":
        np.random.seed(seed + 1000)
        rv = truncnorm(-w_max, w_max, loc=mean, scale=std)
        return rv.rvs(size=time_hor)
    if distr == "uniform":
        np.random.seed(seed + 1000)
        return np.random.uniform(low=lb, high=ub, size=time_hor)
    if distr == "bernouli":
        np.random.seed(seed + 1000)
        r = np.random.rand(time_hor)
        return np.where(r < 0.5, w_max, -w_max)
    raise ValueError(f"unsupported disturbance distribution: {distr!r}")


@dataclass
class NonAnalyticTrajectoryData:
    x: np.ndarray
    u: np.ndarray
    w: np.ndarray
    x_next: np.ndarray
    phi_rows: np.ndarray
    fx: np.ndarray
    fu: np.ndarray


@dataclass
class NonAnalyticDynamics:
    """Simulate x_{t+1} = a_* f(x_t) + b_* f(u_t) + w_t with scalar state."""

    a_star: float
    b_star: float
    distr_w: str = "uniform"
    input_type: str = "uniform"

    def simulate(
        self,
        *,
        time_hor: int,
        seed_w: int,
        seed_u: int,
        x0: float = 0.0,
        param_u: Sequence[float] = (-1.0, 1.0),
        param_w: Sequence[float] = (-0.1, 0.1),
        mult_u: Sequence[float] = (1.0,),
        mult_w: Sequence[float] = (1.0,),
        u_min_f: float = 0.1,
    ) -> NonAnalyticTrajectoryData:
        if self.input_type == "trunc_guass":
            u_max = float(param_u[2])
        else:
            u_max = 1.0

        if self.distr_w == "trunc_guass":
            w_max = float(param_w[2])
        else:
            w_max = 0.1

        u_explore = generate_u(
            self.input_type,
            time_hor,
            seed_u,
            mean=param_u[0],
            std=param_u[1] if len(param_u) > 1 else 0.0,
            u_max=u_max,
            lb=param_u[0],
            ub=param_u[1],
        )
        w = mult_w[0] * generate_w(
            self.distr_w,
            time_hor,
            seed_w,
            mean=param_w[0] if len(param_w) > 2 else 0.0,
            std=param_w[1] if len(param_w) > 2 else 0.0,
            w_max=w_max,
            lb=param_w[0],
            ub=param_w[1],
        )

        x = np.zeros(time_hor + 1, dtype=float)
        u = np.zeros(time_hor, dtype=float)
        x[0] = float(x0)

        phi_rows: List[np.ndarray] = []
        fx_vals: List[float] = []
        fu_vals: List[float] = []
        y_vals: List[float] = []

        for t in range(time_hor):
            u_t = float(mult_u[0] * u_explore[t])
            # if u_t <= 0.0:
            #     u_t = abs(u_t) + 0.75
            # u[t] = u_t

            fx_t = float(bump_function(x[t]))
            fu_t = float(bump_function(u_t))
            if fu_t <= u_min_f:
                u_t = 0.75
                u[t] = u_t
                fu_t = float(bump_function(u_t))

            x[t + 1] = self.a_star * fx_t + self.b_star * fu_t + w[t]

            phi_rows.append(phi_features(x[t], u_t))
            fx_vals.append(fx_t)
            fu_vals.append(fu_t)
            y_vals.append(x[t + 1])

        return NonAnalyticTrajectoryData(
            x=x,
            u=u,
            w=w,
            x_next=np.asarray(y_vals, dtype=float),
            phi_rows=np.vstack(phi_rows),
            fx=np.asarray(fx_vals, dtype=float),
            fu=np.asarray(fu_vals, dtype=float),
        )
