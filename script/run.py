"""
run.py – Generates static PDF figures for three conditions × two p_GM values.

Conditions:
  asocial        : social_learning_weight=0, conformity_weight=0
  social_learning: social_learning_weight=1, conformity_weight=0
  conformity     : social_learning_weight=0, conformity_weight=1

Growth proportions: 0.2 (GM as minority) and 0.8 (GM as majority)

Output structure:
  output/{figure_type}/pGM{pGM_value}/{condition}.pdf
  output/fig_{figure_type}.pdf   <- composite (row1=low pGM, row2=high pGM)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(__file__))
from model import MindsetModel

# ── shared defaults ────────────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    width=50, height=50,
    reward_slope=1.0, reward_intercept=10.0,
    time_horizon=50, malleability=0.5,
    growth_prior_mean=0.8, growth_prior_strength=20,
    social_learning_weight=0.0,
    conformity_weight=0.0,
    seed=42,
)

CONDITIONS = {
    "asocial":         {},
    "social_learning": {"social_learning_weight": 1.0},
    "conformity":      {"conformity_weight": 1.0},
    "both_social":     {"social_learning_weight": 0.5, "conformity_weight": 0.5},
}

CONDITION_LABELS = {
    "asocial":         "Asocial",
    "social_learning": "Social Learning",
    "conformity":      "Conformity",
    "both_social":     "Social Learning + Conformity",
}

GM_PROPORTIONS = [0.2, 0.8]
SNAPSHOT_STEP  = 10

COLOR_GROWTH    = "#2E6DA4"
COLOR_FIXED     = "#C0392B"
COLOR_CULTIVATE = "#222222"
COLOR_HARVEST   = "#bEbEbE"

DPI = 150


# ── helpers ───────────────────────────────────────────────────────────────────

def _optimal_switch(model):
    D = model.time_horizon
    w = model.reward_slope
    R = model.reward_intercept
    m = model.malleability
    d_tilde = D / 2 - R / (2 * w * m)
    return int(np.clip(round(d_tilde), 0, D))


def run_model_with_snapshots(params, snapshot_step):
    m = MindsetModel(**params)
    snap_action = snap_belief = None
    while m.running:
        m.step()
        if m.time_step == snapshot_step:
            W, H = m.grid.width, m.grid.height
            action_arr = np.full((H, W), np.nan)
            belief_arr = np.full((H, W), np.nan)
            for agent in m.agents:
                x, y = agent.pos
                action_arr[y, x] = (
                    1 if (agent.choice_history and agent.choice_history[-1] == 'cultivate')
                    else 0
                )
                belief_arr[y, x] = agent.malleability_estimate
            snap_action = action_arr.copy()
            snap_belief = belief_arr.copy()
    df = m.datacollector.get_model_vars_dataframe()
    return m, df, snap_action, snap_belief


# ── per-panel plotters ────────────────────────────────────────────────────────

def plot_action_trajectory(ax, df, model, title, show_ylabel=True, show_xlabel=True):
    switch = _optimal_switch(model)
    D = model.time_horizon
    steps = df.index + 1
    if switch > 0:
        ax.axvspan(0, switch, color="#e6e6e6", alpha=0.6, zorder=0)
    ax.axhline(model.growth_proportion, color="#6B6B6B", linestyle="--", linewidth=1)
    ax.plot(steps, df["cultivate_proportion"], color=COLOR_CULTIVATE, linewidth=2.5, zorder=2)
    ax.set_xlim(0, D)
    ax.set_ylim(0, 1)
    if show_xlabel:
        ax.set_xlabel("Step", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Cultivate proportion", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)


def plot_belief_trajectory(ax, df, model, title, show_ylabel=True, show_xlabel=True):
    switch = _optimal_switch(model)
    D = model.time_horizon
    steps = df.index + 1
    if switch > 0:
        ax.axvspan(0, switch, color="#e6e6e6", alpha=0.6, zorder=0)
    ax.axhline(model.malleability, color="#6B6B6B", linestyle="--", linewidth=1)
    for mindset, color in [("growth", COLOR_GROWTH), ("fixed", COLOR_FIXED)]:
        mean = df[f"{mindset}_belief_mean"]
        std  = df[f"{mindset}_belief_std"]
        ax.plot(steps, mean, color=color, linewidth=2.5, label=f"{mindset}", zorder=2)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, zorder=1)
    ax.set_xlim(0, D)
    ax.set_ylim(0, 1)
    if show_xlabel:
        ax.set_xlabel("Step", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Malleability estimate", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=8)


def plot_action_grid(ax, snap, title):
    cmap = mcolors.ListedColormap([COLOR_HARVEST, COLOR_CULTIVATE])
    ax.imshow(snap, vmin=0, vmax=1, cmap=cmap, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight="bold")
    legend_elems = [
        Patch(facecolor=COLOR_CULTIVATE, label="Cultivate"),
        Patch(facecolor=COLOR_HARVEST,   label="Harvest"),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=7)


def plot_belief_grid(ax, snap, model, title):
    im = ax.imshow(snap, vmin=0, vmax=1, cmap="RdBu", origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cond_list = list(CONDITIONS.keys())

    # create output dirs
    os.makedirs("../output", exist_ok=True)
    for fig_type in ["action_trajectory", "belief_trajectory", "action_grid", "belief_grid"]:
        for pgm in GM_PROPORTIONS:
            os.makedirs(f"../output/{fig_type}/pGM{pgm}", exist_ok=True)

    # run all conditions x proportions
    results = {}  # key: (pgm, cond)
    for pgm in GM_PROPORTIONS:
        for cond, overrides in CONDITIONS.items():
            params = {**DEFAULT_PARAMS, "growth_proportion": pgm, **overrides}
            print(f"Running pGM={pgm}, condition={cond} ...")
            m, df, snap_action, snap_belief = run_model_with_snapshots(params, SNAPSHOT_STEP)
            results[(pgm, cond)] = dict(model=m, df=df,
                                        snap_action=snap_action,
                                        snap_belief=snap_belief)

    # ── individual PDFs ───────────────────────────────────────────────────────
    for pgm in GM_PROPORTIONS:
        for cond in cond_list:
            res = results[(pgm, cond)]
            m, df = res["model"], res["df"]
            label = CONDITION_LABELS[cond]

            fig, ax = plt.subplots(figsize=(4, 3), dpi=DPI)
            plot_action_trajectory(ax, df, m, label)
            fig.tight_layout()
            fig.savefig(f"../output/action_trajectory/pGM{pgm}/{cond}.pdf", dpi=DPI)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4, 3), dpi=DPI)
            plot_belief_trajectory(ax, df, m, label)
            fig.tight_layout()
            fig.savefig(f"../output/belief_trajectory/pGM{pgm}/{cond}.pdf", dpi=DPI)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4, 4), dpi=DPI)
            plot_action_grid(ax, res["snap_action"], label)
            fig.tight_layout()
            fig.savefig(f"../output/action_grid/pGM{pgm}/{cond}.pdf", dpi=DPI)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4, 4), dpi=DPI)
            plot_belief_grid(ax, res["snap_belief"], m, label)
            fig.tight_layout()
            fig.savefig(f"../output/belief_grid/pGM{pgm}/{cond}.pdf", dpi=DPI)
            plt.close(fig)

    print("Individual PDFs saved.")

    # ── composite figures: 2 rows x 3 cols ───────────────────────────────────
    row_labels = {
        0.2: r"$p_{\mathrm{GM}}=0.2$" + "\n(GM as minority)",
        0.8: r"$p_{\mathrm{GM}}=0.8$" + "\n(GM as majority)",
    }

    specs = [
        ("action_trajectory", plot_action_trajectory, (17, 6.5)),
        ("belief_trajectory", plot_belief_trajectory, (17, 6.5)),
        ("action_grid",       None,                   (17, 9.0)),
        ("belief_grid",       None,                   (17, 9.0)),
    ]

    for fig_type, _, figsize in specs:
        fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=DPI)

        for row_i, pgm in enumerate(GM_PROPORTIONS):
            for col_i, cond in enumerate(cond_list):
                ax = axes[row_i, col_i]
                res = results[(pgm, cond)]
                title = CONDITION_LABELS[cond]
                show_ylabel = (col_i == 0)
                show_xlabel = (row_i == 1)

                if fig_type == "action_trajectory":
                    plot_action_trajectory(ax, res["df"], res["model"], title,
                                           show_ylabel=show_ylabel, show_xlabel=show_xlabel)
                elif fig_type == "belief_trajectory":
                    plot_belief_trajectory(ax, res["df"], res["model"], title,
                                           show_ylabel=show_ylabel, show_xlabel=show_xlabel)
                elif fig_type == "action_grid":
                    plot_action_grid(ax, res["snap_action"], title)
                elif fig_type == "belief_grid":
                    plot_belief_grid(ax, res["snap_belief"], res["model"], title)

                # row annotation on leftmost panel
                if col_i == 0:
                    base_ylabel = ax.get_ylabel()
                    new_label = row_labels[pgm]
                    if base_ylabel:
                        new_label = base_ylabel + "\n" + new_label
                    ax.set_ylabel(new_label, fontsize=9)

        fig.tight_layout()
        out_path = f"../output/fig_{fig_type}.pdf"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Composite saved: {out_path}")

    print("All done. Output in:", os.path.abspath("../output/"))


if __name__ == "__main__":
    main()