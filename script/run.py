"""
run.py – Generates MP4 animations for three experimental conditions:
  1. social_learning  : social_learning_weight=1, conformity_weight=0 (default)
  2. control          : social_learning_weight=0 (default), conformity_weight=0 (default)
  3. conformity       : conformity_weight=1, social_learning_weight=0 (default)

For each condition four animations are saved:
  {condition}_action_grid.mp4
  {condition}_belief_grid.mp4
  {condition}_cultivation_plot.mp4
  {condition}_belief_plot.mp4
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ── ensure local modules resolve ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from model import MindsetModel

# ── shared defaults ────────────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    width=50, height=50,
    reward_slope=1.0, reward_intercept=10.0,
    time_horizon=50, malleability=0.5,
    growth_proportion=0.7,
    growth_prior_mean=0.8, growth_prior_strength=20,
    social_learning_weight=0.0,
    conformity_weight=0.0,
    seed=42,
)

CONDITIONS = {
    "social_learning": {"social_learning_weight": 1.0},
    "control":         {},
    "conformity":      {"conformity_weight": 1.0},
}

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 10
DPI = 100


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_model(params: dict) -> MindsetModel:
    """Run the model to completion and return it (with datacollector populated)."""
    m = MindsetModel(**params)
    while m.running:
        m.step()
    return m


def _agent_grid(model, attr_fn):
    """Return a (height x width) array of per-agent values."""
    grid = np.full((model.grid.height, model.grid.width), np.nan)
    for agent in model.agents:
        x, y = agent.pos
        grid[y, x] = attr_fn(agent)
    return grid


def _optimal_switch(model) -> int:
    D = model.time_horizon
    w = model.reward_slope
    R = model.reward_intercept
    m = model.malleability
    d_tilde = D / 2 - R / (2 * w * m)
    return int(np.clip(round(d_tilde), 0, D))


# ─────────────────────────────────────────────────────────────────────────────
# per-frame data builders
# ─────────────────────────────────────────────────────────────────────────────

def build_frame_data(params: dict):
    """
    Run the model step-by-step, capturing per-frame state.
    Returns a list of dicts (one per step).
    """
    m = MindsetModel(**params)
    frames = []
    while m.running:
        m.step()
        # snapshot agent states
        action_arr = _agent_grid(
            m, lambda a: 1 if (a.choice_history and a.choice_history[-1] == 'cultivate') else 0)
        belief_arr = _agent_grid(m, lambda a: a.malleability_estimate)
        df = m.datacollector.get_model_vars_dataframe()
        frames.append(dict(
            step=m.time_step,
            action_arr=action_arr.copy(),
            belief_arr=belief_arr.copy(),
            df=df.copy(),
        ))
    return frames, m


# ─────────────────────────────────────────────────────────────────────────────
# animation builders
# ─────────────────────────────────────────────────────────────────────────────

def make_action_grid_anim(frames, model, path):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=DPI)
    im = ax.imshow(frames[0]["action_arr"], vmin=0, vmax=1,
                   cmap=plt.cm.colors.ListedColormap(["#bEbEbE", "#222222"]),
                   origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    legend_elems = [Patch(facecolor="#222222", label="Cultivate"),
                    Patch(facecolor="#bEbEbE", label="Harvest")]
    ax.legend(handles=legend_elems, loc="upper right", fontsize=8)
    title = ax.set_title("Step 0", fontsize=10)

    def update(i):
        im.set_data(frames[i]["action_arr"])
        title.set_text(f"Step {frames[i]['step']}")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000//FPS, blit=True)
    ani.save(path, writer="ffmpeg", fps=FPS)
    plt.close(fig)
    print(f"  saved {path}")


def make_belief_grid_anim(frames, model, path):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=DPI)
    im = ax.imshow(frames[0]["belief_arr"], vmin=0, vmax=1,
                   cmap="RdBu", origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Malleability estimate", fontsize=8)
    title = ax.set_title("Step 0", fontsize=10)

    def update(i):
        im.set_data(frames[i]["belief_arr"])
        title.set_text(f"Step {frames[i]['step']}")
        return [im, title]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000//FPS, blit=False)
    ani.save(path, writer="ffmpeg", fps=FPS)
    plt.close(fig)
    print(f"  saved {path}")


def make_cultivation_plot_anim(frames, model, path):
    D = model.time_horizon
    switch = _optimal_switch(model)

    fig, ax = plt.subplots(figsize=(5.4, 4), dpi=DPI)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.94)
    if switch > 0:
        ax.axvspan(0, switch, color="#e6e6e6", alpha=0.5, zorder=0)
    ax.axhline(model.growth_proportion, color="#6B6B6B", linestyle="--", linewidth=1)
    ax.set_xlim(0, D); ax.set_ylim(0, 1)
    ax.set_xlabel("Step"); ax.set_ylabel("Cultivate proportion")
    line, = ax.plot([], [], color="#222222", linewidth=3, zorder=2)
    title = ax.set_title("Step 0", fontsize=10)

    def update(i):
        df = frames[i]["df"]
        steps = df.index + 1
        line.set_data(steps, df["cultivate_proportion"])
        title.set_text(f"Step {frames[i]['step']}")
        return [line, title]

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000//FPS, blit=True)
    ani.save(path, writer="ffmpeg", fps=FPS)
    plt.close(fig)
    print(f"  saved {path}")


def make_belief_plot_anim(frames, model, path):
    D = model.time_horizon
    switch = _optimal_switch(model)

    fig, ax = plt.subplots(figsize=(5.4, 4), dpi=DPI)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.94)
    if switch > 0:
        ax.axvspan(0, switch, color="#e6e6e6", alpha=0.5, zorder=0)
    ax.axhline(model.malleability, color="#6B6B6B", linestyle="--", linewidth=1)
    ax.set_xlim(0, D); ax.set_ylim(0, 1)
    ax.set_xlabel("Step"); ax.set_ylabel("Malleability estimate")

    COLORS = {"growth": "#2E6DA4", "fixed": "#C0392B"}
    lines = {}
    fills = {}
    for mindset, color in COLORS.items():
        lines[mindset], = ax.plot([], [], color=color, linewidth=3, label=f"{mindset} mindset", zorder=2)
        fills[mindset] = ax.fill_between([], [], [], color=color, alpha=0.15, zorder=1)
    ax.legend(fontsize=8)
    title = ax.set_title("Step 0", fontsize=10)

    def update(i):
        df = frames[i]["df"]
        steps = df.index + 1
        artists = [title]
        for mindset in COLORS:
            mean = df[f"{mindset}_belief_mean"]
            std  = df[f"{mindset}_belief_std"]
            lines[mindset].set_data(steps, mean)
            # re-draw fill_between by replacing collection
            fills[mindset].remove()
            fills[mindset] = ax.fill_between(
                steps, mean - std, mean + std, color=COLORS[mindset], alpha=0.15, zorder=1)
            artists.append(lines[mindset])
        title.set_text(f"Step {frames[i]['step']}")
        return artists

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000//FPS, blit=False)
    ani.save(path, writer="ffmpeg", fps=FPS)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    for cond_name, overrides in CONDITIONS.items():
        params = {**DEFAULT_PARAMS, **overrides}
        print(f"\n{'='*60}")
        print(f"Condition: {cond_name}  |  overrides: {overrides or '(none)'}")
        print(f"{'='*60}")

        print("  Running simulation …")
        frames, model = build_frame_data(params)
        print(f"  Completed {len(frames)} steps.")

        base = os.path.join(OUTPUT_DIR, cond_name)

        print("  Rendering action grid …")
        make_action_grid_anim(frames, model, f"{base}_action_grid.mp4")

        print("  Rendering belief grid …")
        make_belief_grid_anim(frames, model, f"{base}_belief_grid.mp4")

        print("  Rendering cultivation plot …")
        make_cultivation_plot_anim(frames, model, f"{base}_cultivation_plot.mp4")

        print("  Rendering belief plot …")
        make_belief_plot_anim(frames, model, f"{base}_belief_plot.mp4")

    print("\nAll done! Files written to:", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()