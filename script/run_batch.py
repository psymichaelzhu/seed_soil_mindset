"""
run_batch.py – Runs N_SEEDS simulations per condition × p_GM, saves data,
               and generates batch figures (trajectory plots only).

Conditions:
  asocial        : social_learning_weight=0, conformity_weight=0
  social_learning: social_learning_weight=1, conformity_weight=0
  conformity     : social_learning_weight=0, conformity_weight=1
  both_social    : social_learning_weight=0.5, conformity_weight=0.5

Growth proportions: 0.2 (GM as minority) and 0.8 (GM as majority)

Data structure:
  data/pGM{pGM_value}_{condition}.pkl  <- list of DataFrames + model_params

Output structure:
  output_batch/{figure_type}/pGM{pGM_value}/{condition}.pdf
  output_batch/fig_{figure_type}.pdf   <- composite (row1=low pGM, row2=high pGM)

Each trajectory plot shows:
  - one thin transparent line per simulation run
  - one opaque line for the cross-run mean
"""

import os
import sys
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import MindsetModel

# ── shared defaults ────────────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    width=20, height=20,
    reward_slope=1.0, reward_intercept=10.0,
    time_horizon=50, malleability=0.5,
    growth_prior_mean=0.8, growth_prior_strength=20,
    social_learning_weight=0.0,
    conformity_weight=0.0,
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
N_SEEDS        = 50
OVERWRITE      = False   # if False, load existing .pkl instead of re-running

COLOR_GROWTH    = "#2E6DA4"
COLOR_FIXED     = "#C0392B"
COLOR_CULTIVATE = "#222222"

DPI = 150


# ── helpers ───────────────────────────────────────────────────────────────────

def _optimal_switch(model_params):
    D = model_params["time_horizon"]
    w = model_params["reward_slope"]
    R = model_params["reward_intercept"]
    m = model_params["malleability"]
    d_tilde = D / 2 - R / (2 * w * m)
    return int(np.clip(round(d_tilde), 0, D))


def run_single(params):
    m = MindsetModel(**params)
    while m.running:
        m.step()
    df = m.datacollector.get_model_vars_dataframe()
    return df


def run_batch(params_base, n_seeds):
    """Run n_seeds simulations, return list of DataFrames."""
    dfs = []
    for seed in tqdm(range(n_seeds)):
        params = {**params_base, "seed": seed}
        df = run_single(params)
        dfs.append(df)
    return dfs


# ── per-panel plotters ────────────────────────────────────────────────────────

def plot_action_trajectory_batch(ax, dfs, model_params, title,
                                 show_ylabel=True, show_xlabel=True):
    switch = _optimal_switch(model_params)
    D      = model_params["time_horizon"]
    steps  = dfs[0].index + 1

    if switch > 0:
        ax.axvspan(0, switch, color="#e6e6e6", alpha=0.6, zorder=0)
    ax.axhline(model_params["growth_proportion"], color="#6B6B6B",
               linestyle="--", linewidth=1)

    # individual runs
    for df in dfs:
        ax.plot(steps, df["cultivate_proportion"],
                color=COLOR_CULTIVATE, linewidth=2, alpha=0.08, zorder=1)

    # cross-run mean
    mean_vals = np.mean([df["cultivate_proportion"].values for df in dfs], axis=0)
    ax.plot(steps, mean_vals, color=COLOR_CULTIVATE, linewidth=2, zorder=2)

    ax.set_xlim(0, D)
    ax.set_ylim(0, 1)
    if show_xlabel:
        ax.set_xlabel("Step", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Cultivate proportion", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.tick_params(labelsize=8)


def plot_belief_trajectory_batch(ax, dfs, model_params, title,
                                 show_ylabel=True, show_xlabel=True):
    switch = _optimal_switch(model_params)
    D      = model_params["time_horizon"]
    steps  = dfs[0].index + 1

    if switch > 0:
        ax.axvspan(0, switch, color="#e6e6e6", alpha=0.6, zorder=0)
    ax.axhline(model_params["malleability"], color="#6B6B6B",
               linestyle="--", linewidth=1)

    for mindset, color in [("growth", COLOR_GROWTH), ("fixed", COLOR_FIXED)]:
        col = f"{mindset}_belief_mean"

        # individual runs (no fill_between, just the mean line per run)
        for df in dfs:
            ax.plot(steps, df[col],
                    color=color, linewidth=2, alpha=0.08, zorder=1)

        # cross-run mean
        mean_vals = np.mean([df[col].values for df in dfs], axis=0)
        ax.plot(steps, mean_vals, color=color, linewidth=2,
                label=mindset, zorder=2)

    ax.set_xlim(0, D)
    ax.set_ylim(0, 1)
    if show_xlabel:
        ax.set_xlabel("Step", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Malleability estimate", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.tick_params(labelsize=8)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    cond_list = list(CONDITIONS.keys())

    # create dirs
    os.makedirs("../data", exist_ok=True)
    os.makedirs("../output_batch", exist_ok=True)
    for fig_type in ["action_trajectory", "belief_trajectory"]:
        for pgm in GM_PROPORTIONS:
            os.makedirs(f"../output_batch/{fig_type}/pGM{pgm}", exist_ok=True)

    # ── run all conditions x proportions and save data ──────────────────────
    results = {}  # key: (pgm, cond)
    for pgm in GM_PROPORTIONS:
        for cond, overrides in CONDITIONS.items():
            pkl_path = f"../data/pGM{pgm}_{cond}.pkl"

            if not OVERWRITE and os.path.exists(pkl_path):
                print(f"Loading existing data: {pkl_path}")
                with open(pkl_path, "rb") as f:
                    data = pickle.load(f)
                dfs          = data["dfs"]
                model_params = data["model_params"]
            else:
                params_base = {**DEFAULT_PARAMS, "growth_proportion": pgm, **overrides}
                print(f"Running {N_SEEDS} seeds: pGM={pgm}, condition={cond} ...")
                dfs = run_batch(params_base, N_SEEDS)

                model_params = {
                    "time_horizon":      params_base["time_horizon"],
                    "reward_slope":      params_base["reward_slope"],
                    "reward_intercept":  params_base["reward_intercept"],
                    "malleability":      params_base["malleability"],
                    "growth_proportion": pgm,
                }

                # save to disk
                with open(pkl_path, "wb") as f:
                    pickle.dump({"dfs": dfs, "model_params": model_params}, f)
                print(f"  Saved: {pkl_path}")

            results[(pgm, cond)] = dict(dfs=dfs, model_params=model_params)

    # ── individual PDFs ───────────────────────────────────────────────────────
    for pgm in GM_PROPORTIONS:
        for cond in cond_list:
            res   = results[(pgm, cond)]
            dfs   = res["dfs"]
            mp    = res["model_params"]
            label = CONDITION_LABELS[cond]

            fig, ax = plt.subplots(figsize=(4, 3), dpi=DPI)
            plot_action_trajectory_batch(ax, dfs, mp, label)
            fig.tight_layout()
            fig.savefig(f"../output_batch/action_trajectory/pGM{pgm}/{cond}.pdf", dpi=DPI)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(4, 3), dpi=DPI)
            plot_belief_trajectory_batch(ax, dfs, mp, label)
            fig.tight_layout()
            fig.savefig(f"../output_batch/belief_trajectory/pGM{pgm}/{cond}.pdf", dpi=DPI)
            plt.close(fig)

    print("Individual PDFs saved.")

    # ── composite figures: 2 rows × 4 cols ───────────────────────────────────
    row_labels = {
        0.2: r"$p_{\mathrm{GM}}=0.2$" + "\n(GM as minority)",
        0.8: r"$p_{\mathrm{GM}}=0.8$" + "\n(GM as majority)",
    }

    specs = [
        ("action_trajectory", (17, 6.5)),
        ("belief_trajectory", (17, 6.5)),
    ]

    for fig_type, figsize in specs:
        fig, axes = plt.subplots(2, 4, figsize=figsize, dpi=DPI)

        for row_i, pgm in enumerate(GM_PROPORTIONS):
            for col_i, cond in enumerate(cond_list):
                ax    = axes[row_i, col_i]
                res   = results[(pgm, cond)]
                title = CONDITION_LABELS[cond]
                show_ylabel = (col_i == 0)
                show_xlabel = (row_i == 1)

                if fig_type == "action_trajectory":
                    plot_action_trajectory_batch(ax, res["dfs"], res["model_params"],
                                                 title, show_ylabel=show_ylabel,
                                                 show_xlabel=show_xlabel)
                elif fig_type == "belief_trajectory":
                    plot_belief_trajectory_batch(ax, res["dfs"], res["model_params"],
                                                 title, show_ylabel=show_ylabel,
                                                 show_xlabel=show_xlabel)

                # row annotation on leftmost panel
                if col_i == 0:
                    base_ylabel = ax.get_ylabel()
                    new_label   = row_labels[pgm]
                    if base_ylabel:
                        new_label = base_ylabel + "\n" + new_label
                    ax.set_ylabel(new_label, fontsize=9)

        fig.tight_layout()
        out_path = f"../output_batch/fig_batch_{fig_type}.pdf"
        fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"Composite saved: {out_path}")

    print("All done. Output in:", os.path.abspath("../output_batch/"))


if __name__ == "__main__":
    main()