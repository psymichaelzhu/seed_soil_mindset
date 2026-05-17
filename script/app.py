import solara
from model import MindsetModel
from mesa.visualization import (
    SolaraViz,
    make_space_component,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle
from mesa.visualization.utils import update_counter
from matplotlib.figure import Figure
import numpy as np

# ── visualization figure default ──────────────────────────────────────────────────────────
PLOT_FIGSIZE = (5.4, 4)

def format_plot_fig(fig):
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.16, top=0.94)


# ── Agent portrayals ──────────────────────────────────────────────────────────

def action_portrayal(agent):
    color = "#222222" if (agent.choice_history and agent.choice_history[-1] == 'cultivate') else "#bEbEbE"
    return AgentPortrayalStyle(color=color, marker="s", size=75)


def belief_portrayal(agent):
    v = agent.malleability_estimate  # (0, 1)
    # Reverse the belief color mapping:
    if v >= 0.5:
        t = (v - 0.5) / 0.5
        r, g, b = 1.0 - t, 1.0 - t, 1.0   # white → blue
    else:
        t = v / 0.5
        r, g, b = 1.0, t, t               # red → white
    color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
    return AgentPortrayalStyle(color=color, marker="s", size=75)


def clean_ax(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")


# ── Custom cultivation plot ───────────────────────────────────────────────────

@solara.component
def CultivationPlot(model):
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    fig = Figure(figsize=PLOT_FIGSIZE)
    ax = fig.subplots()
    D = model.time_horizon
    w = model.reward_slope
    R = model.reward_intercept
    m = model.malleability
    d_tilde = D / 2 - R / (2 * w * m)
    switch = int(np.clip(round(d_tilde), 0, D))
    # shaded cultivate region (before optimal switch point)
    if switch > 0:
        ax.axvspan(0, switch, color="#e6e6e6", alpha=0.5, zorder=0)
    steps = df.index + 1
    ax.plot(steps, df["cultivate_proportion"], color="#222222", zorder=2,linewidth=3)
    ax.axhline(model.growth_proportion, color="#6B6B6B", linestyle="--", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cultivate proportion")
    ax.set_xlim(0, D)
    ax.set_ylim(0, 1)
    format_plot_fig(fig)
    solara.FigureMatplotlib(fig)


# ── Custom belief trajectory plot ────────────────────────────────────────────

@solara.component
def BeliefPlot(model):
    update_counter.get()
    df = model.datacollector.get_model_vars_dataframe()
    fig = Figure(figsize=PLOT_FIGSIZE)
    ax = fig.subplots()
    D = model.time_horizon
    w = model.reward_slope
    R = model.reward_intercept
    m = model.malleability
    d_tilde = D / 2 - R / (2 * w * m)
    switch = int(np.clip(round(d_tilde), 0, D))
    # shaded cultivate region (before optimal switch point)
    if switch > 0:
        ax.axvspan(0, switch, color="#e6e6e6", alpha=0.5, zorder=0)
    steps = df.index + 1
    # Reverse the belief line colors: growth → blue, fixed → red
    for mindset, color in [("growth", "#2E6DA4"), ("fixed", "#C0392B")]:
        mean = df[f"{mindset}_belief_mean"]
        std  = df[f"{mindset}_belief_std"]
        ax.plot(steps, mean, color=color, label=f"{mindset} mindset", zorder=2,linewidth=3)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15, zorder=1)
    ax.axhline(model.malleability, color="#6B6B6B", linestyle="--", linewidth=1)
    ax.set_xlabel("Step")
    ax.set_ylabel("Malleability estimate")
    ax.set_xlim(0, D)
    ax.set_ylim(0, 1)
    ax.legend()
    format_plot_fig(fig)
    solara.FigureMatplotlib(fig)


# ── Model parameters (sliders) ────────────────────────────────────────────────

model_params = {
    "growth_proportion": {
        "type": "SliderFloat", "value": 0.3,
        "label": "Growth mindset proportion", "min": 0, "max": 1, "step": 0.01,
    },
    "growth_prior_mean": {
        "type": "SliderFloat", "value": 0.8,
        "label": "Growth prior mean", "min": 0.01, "max": 0.99, "step": 0.01,
    },
    "growth_prior_strength": {
        "type": "SliderInt", "value": 20,
        "label": "Growth prior strength", "min": 1, "max": 100, "step": 1,
    },
    "time_horizon": {
        "type": "SliderInt", "value": 50,
        "label": "Time horizon", "min": 5, "max": 100, "step": 1,
    },
    "reward_slope": {
        "type": "SliderFloat", "value": 1.0,
        "label": "Reward slope (w)", "min": 0.1, "max": 5, "step": 0.1,
    },
    "reward_intercept": {
        "type": "SliderFloat", "value": 10,
        "label": "Reward intercept (R)", "min": 0, "max": 20, "step": 0.5,
    },
    "malleability": {
        "type": "SliderFloat", "value": 0.5,
        "label": "True malleability (m)", "min": 0.01, "max": 0.99, "step": 0.01,
    },
    "social_learning_weight": {
        "type": "SliderFloat", "value": 0,
        "label": "Social learning weight", "min": 0, "max": 1, "step": 0.01,
    },
    "conformity_weight": {
        "type": "SliderFloat", "value": 0,
        "label": "Conformity weight", "min": 0, "max": 1, "step": 0.01,
    },
    "width": {
        "type": "SliderInt", "value": 50,
        "label": "Grid width", "min": 5, "max": 100, "step": 1,
    },
    "height": {
        "type": "SliderInt", "value": 50,
        "label": "Grid height", "min": 5, "max": 100, "step": 1,
    },
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
}

# ── Visualisation components ──────────────────────────────────────────────────

initial_params = {
    name: spec["value"]
    for name, spec in model_params.items()
}

mindset_model = MindsetModel(**initial_params)

ActionGrid = make_space_component(action_portrayal, draw_grid=False, post_process=clean_ax)
BeliefGrid = make_space_component(belief_portrayal, draw_grid=False, post_process=clean_ax)

page = SolaraViz(
    mindset_model,
    components=[ActionGrid, BeliefGrid, CultivationPlot, BeliefPlot],
    model_params=model_params,
    name="Mindset Model",
)
page