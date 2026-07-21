"""Figure 1-style parameter-plane plots and diameter curves."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

from set_membership_non_analytic import polygon_plot_points
from lse_confidence import (
    LSEConfidenceRegion,
    confidence_boundary_points,
    lse_confidence_region,
)
from sme_pendulum_polytope import (
    sme_directional_diameter,
    sme_polytope_vertices,
)

COLOR_SME = (0.9290, 0.6940, 0.1250)
COLOR_LSE = (0.0, 0.4470, 0.7410)
COLOR_ACTIVE = (0.4660, 0.6740, 0.1880)
COLOR_TRUE = "red"

PLOT_FONT = 26
LINEWIDTH = 5.0
LEGEND_ORDER = ("LSE", "SME", "Active", "True")
LEGEND_KW = dict(
    loc="upper right",
    frameon=False,
    labelspacing=0.12,
    handletextpad=0.35,
    borderaxespad=0.15,
    handlelength=0.9,
    borderpad=0.2,
)


def _compact_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(
        [by_label[k] for k in LEGEND_ORDER if k in by_label],
        [k for k in LEGEND_ORDER if k in by_label],
        **LEGEND_KW,
    )


def apply_paper_style() -> None:
    plt.rc("font", size=PLOT_FONT)
    plt.rc("axes", titlesize=PLOT_FONT, labelsize=PLOT_FONT)
    plt.rc("xtick", labelsize=PLOT_FONT)
    plt.rc("ytick", labelsize=PLOT_FONT)
    plt.rc("legend", fontsize=PLOT_FONT)
    plt.rc("figure", titlesize=PLOT_FONT)
    plt.rcParams.update({"axes.grid": True, "axes.edgecolor": "black", "axes.linewidth": 2})


def _active_samples_at_T(
    T: int,
    warmup: int,
    phi_passive: np.ndarray,
    y_passive: np.ndarray,
    phi_active: np.ndarray,
    y_active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Before the first active episode (T <= warmup), use passive PD data for active LSE."""
    if T <= warmup:
        return phi_passive[:T], y_passive[:T]
    return phi_active[:T], y_active[:T]


def _axis_limits(
    sme_vertices: np.ndarray,
    theta_star: np.ndarray,
    passive: LSEConfidenceRegion,
    active: LSEConfidenceRegion,
    *,
    margin: float = 0.08,
) -> tuple[tuple[float, float], tuple[float, float]]:
    chunks = [
        np.asarray(sme_vertices, float),
        np.asarray(theta_star, float).reshape(1, -1),
        confidence_boundary_points(passive.center, passive.M, passive.beta),
        confidence_boundary_points(active.center, active.M, active.beta),
    ]
    pts = np.vstack([c for c in chunks if c.size > 0])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = np.maximum(hi - lo, 0.05 * np.maximum(np.abs(hi), 1.0))
    return (
        (float(lo[0] - margin * span[0]), float(hi[0] + margin * span[0])),
        (float(lo[1] - margin * span[1]), float(hi[1] + margin * span[1])),
    )


def sme_set_diameter(
    y: np.ndarray,
    phi: np.ndarray,
    w_max: float,
    *,
    vertices: Optional[np.ndarray] = None,
) -> float:
    """SME diameter: max directional width (matches vertex span for the polytope)."""
    d_dir = sme_directional_diameter(y, phi, w_max)
    if vertices is None:
        vertices, _ = sme_polytope_vertices(y, phi, w_max)
    if vertices.shape[0] >= 2:
        d_vert = float(np.max(pdist(vertices)))
        return max(d_dir, d_vert)
    return d_dir


def plot_parameter_panel(
    ax,
    *,
    sme_vertices: np.ndarray,
    passive: LSEConfidenceRegion,
    active: LSEConfidenceRegion,
    theta_star: np.ndarray,
    title: str,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    show_legend: bool = False,
) -> None:
    """Draw LSE → Active → SME → True (later layers on top)."""
    for region, color, label, zorder in (
        (passive, COLOR_LSE, "LSE", 1),
        (active, COLOR_ACTIVE, "Active", 2),
    ):
        pts = confidence_boundary_points(region.center, region.M, region.beta)
        ax.fill(
            pts[:, 0], pts[:, 1], color=color, alpha=0.35, edgecolor=color,
            linewidth=1.5, label=label, zorder=zorder,
        )
    poly = polygon_plot_points(sme_vertices)
    if poly.shape[0] > 0:
        ax.fill(
            poly[:, 0], poly[:, 1], color=COLOR_SME, alpha=0.85, edgecolor=COLOR_SME,
            linewidth=1.5, label="SME", zorder=3,
        )
    ax.plot(
        theta_star[0], theta_star[1], marker="*", color=COLOR_TRUE,
        ms=7, mew=1.2, label="True", zorder=4, linestyle="None",
    )
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(title)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    if show_legend:
        _compact_legend(ax)


def plot_diameter_panel(
    ax,
    T_values: Sequence[int],
    sme_d: np.ndarray,
    lse_passive_d: np.ndarray,
    lse_active_d: np.ndarray,
) -> None:
    ax.semilogy(T_values, sme_d, color=COLOR_SME, lw=LINEWIDTH, label="SME")
    ax.semilogy(T_values, lse_passive_d, color=COLOR_LSE, lw=LINEWIDTH, label="LSE")
    ax.semilogy(T_values, lse_active_d, color=COLOR_ACTIVE, lw=LINEWIDTH, label="Active")
    ax.set_xlabel("$T$")
    ax.set_ylabel("Diameter")
    ax.set_title("Diameter vs $T$")
    _compact_legend(ax)


def compute_diameter_curves(
    T_values: Iterable[int],
    phi_passive: np.ndarray,
    y_passive: np.ndarray,
    phi_active: np.ndarray,
    y_active: np.ndarray,
    *,
    w_max: float,
    lse_kw: dict,
    warmup: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sme_d, lse_p, lse_a = [], [], []
    for T in T_values:
        y_p, z_p = y_passive[:T], phi_passive[:T]
        z_a, y_a = _active_samples_at_T(T, warmup, phi_passive, y_passive, phi_active, y_active)
        sme_d.append(sme_set_diameter(y_p, z_p, w_max))
        lse_p.append(lse_confidence_region(z_p, y_p, **lse_kw).diameter)
        lse_a.append(lse_confidence_region(z_a, y_a, **lse_kw).diameter)
    return np.asarray(sme_d), np.asarray(lse_p), np.asarray(lse_a)


def regions_at_T(
    T: int,
    phi_passive: np.ndarray,
    y_passive: np.ndarray,
    phi_active: np.ndarray,
    y_active: np.ndarray,
    *,
    w_max: float,
    lse_kw: dict,
    warmup: int = 0,
) -> tuple[np.ndarray, LSEConfidenceRegion, LSEConfidenceRegion]:
    y_p, z_p = y_passive[:T], phi_passive[:T]
    z_a, y_a = _active_samples_at_T(T, warmup, phi_passive, y_passive, phi_active, y_active)
    sme_v, _ = sme_polytope_vertices(y_p, z_p, w_max)
    passive = lse_confidence_region(z_p, y_p, **lse_kw)
    active = lse_confidence_region(z_a, y_a, **lse_kw)
    return sme_v, passive, active


def make_figure(
    T_left: int,
    T_right: int,
    *,
    sme_left: np.ndarray,
    passive_left: LSEConfidenceRegion,
    active_left: LSEConfidenceRegion,
    sme_right: np.ndarray,
    passive_right: LSEConfidenceRegion,
    active_right: LSEConfidenceRegion,
    T_curve: Sequence[int],
    sme_d: np.ndarray,
    lse_p_d: np.ndarray,
    lse_a_d: np.ndarray,
    theta_star: np.ndarray,
    figsize: tuple[float, float] = (24.0, 8.0),
) -> tuple[plt.Figure, np.ndarray]:
    apply_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    plot_diameter_panel(axes[0], T_curve, sme_d, lse_p_d, lse_a_d)
    xlim, ylim = _axis_limits(sme_left, theta_star, passive_left, active_left)
    xlim_right, ylim_right = _axis_limits(sme_right, theta_star, passive_right, active_right)

    plot_parameter_panel(
        axes[1], sme_vertices=sme_left, passive=passive_left, active=active_left,
        theta_star=theta_star, title=rf"$T = {T_left}$",
        xlim=xlim, ylim=ylim, show_legend=True,
    )
    plot_parameter_panel(
        axes[2], sme_vertices=sme_right, passive=passive_right, active=active_right,
        theta_star=theta_star, title=rf"$T = {T_right}$",
        xlim=xlim_right, ylim=ylim_right,show_legend=True,
    )
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.12, top=0.92, wspace=0.28)
    return fig, axes


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor="white")
