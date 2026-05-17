from mesa import Agent
import numpy as np


def beta_mean(alpha, beta):
    return alpha / (alpha + beta)


# ─────────────────────────────────────────────
# Base agent: knows true malleability, no belief
# ─────────────────────────────────────────────
class FullKnowledgeAgent(Agent):
    """
    Phases:
      plan   → _get_malleability → _estimate_optimal_switch → _plan_action
      act    → _collect_logits → _select → _observe
      update → (none)
    """

    def __init__(self, model):
        super().__init__(model)
        self.skill_level = 0
        self.action_logits = np.zeros(2, dtype=float)
        self.choice_history = []
        self.outcome_history = []
        self.reward_history = []

    # ── plan helpers ──────────────────────────

    def _get_malleability(self, model):
        return model.malleability

    def _estimate_optimal_switch(self, model):
        D = model.time_horizon - model.time_step
        w = model.reward_slope
        R = model.reward_intercept
        m = self._get_malleability(model)
        R_t = R + w * self.skill_level
        d_tilde = D / 2 - R_t / (2 * w * m)
        self.switch_point = int(np.clip(round(d_tilde), 0, D))

    def _plan_action(self):
        vote = np.array([1, 0]) if self.switch_point > 0 else np.array([0, 1])
        self.action_logits += vote

    # ── act helpers ───────────────────────────

    def _collect_logits(self, model):
        """Return list of (logits, weight).
        Called after ALL agents have finished plan(),
        so neighbours' action_logits are fully populated.
        """
        return [(self.action_logits.copy(), 1.0)]

    '''
    def _select(self, model):
        total = np.zeros(2)
        for logits, w in self._collect_logits(model):
            total += logits * w
        probs = total / total.sum()
        self.action = np.random.choice(['cultivate', 'harvest'], p=probs)
        self.action_logits = np.zeros(2)    # reset only after social integration is done
    '''

    def _select(self, model):
        total = np.zeros(2)
        for logits, w in self._collect_logits(model):
            total += logits * w
        self.action = 'cultivate' if total[0] >= total[1] else 'harvest'  # deterministic
        # NOTE: action_logits are NOT reset here; reset_logits() is called
        # by the model after ALL agents have finished select(), so that
        # neighbours' logits remain readable throughout the select phase.

    def _observe(self, model):
        if self.action == 'cultivate':
            outcome = self.cultivation_outcomes[model.time_step]
            if outcome:
                self.skill_level += 1
            reward = 0
        else:  # harvest
            outcome = None
            reward = model.reward_function(self.skill_level)
        self.choice_history.append(self.action)
        self.outcome_history.append(outcome)
        self.reward_history.append(reward)

    # ── Mesa entry points ─────────────────────

    def plan(self, model):
        """Phase ①: private estimation, writes action_logits."""
        self._estimate_optimal_switch(model)
        self._plan_action()

    def select(self, model):
        """Phase ②: read neighbours' logits and decide action (synchronous)."""
        self._select(model)

    def observe(self, model):
        """Phase ③: execute action and record outcome."""
        self._observe(model)

    def reset_logits(self, model):
        """Phase ④a: clear action_logits after all agents have selected."""
        self.action_logits = np.zeros(2)

    def act(self, model):
        """Legacy entry point (kept for compatibility): select + observe."""
        self._select(model)
        self.action_logits = np.zeros(2)
        self._observe(model)

    def update(self, model):
        """Phase ④: no-op for FullKnowledgeAgent."""
        pass


# ─────────────────────────────────────────────
# Mindset agent: unknown malleability, private belief
# ─────────────────────────────────────────────
class MindsetAgent(FullKnowledgeAgent):
    """
    Overrides:
      plan   → _get_malleability returns malleability_estimate
      update → _collect_outcomes + _apply_outcome_to_belief + _estimate_malleability
    Everything else inherited.
    """

    def __init__(self, model, prior):
        super().__init__(model)
        self.belief = list(prior)
        self.malleability_estimate = beta_mean(*self.belief)

    # ── plan override ─────────────────────────

    def _get_malleability(self, model):
        return self.malleability_estimate

    # ── update helpers ────────────────────────

    def _apply_outcome_to_belief(self, outcome, weight=1.0):
        if outcome is True:
            self.belief[0] += weight
        elif outcome is False:
            self.belief[1] += weight
        # outcome is None (harvest) → no update

    def _estimate_malleability(self):
        self.malleability_estimate = beta_mean(self.belief[0], self.belief[1])

    def _collect_outcomes(self, model):
        """Return list of (outcome, weight) to update belief with."""
        return [(self.outcome_history[-1], 1.0)]

    def update(self, model):
        """Phase ④."""
        for outcome, w in self._collect_outcomes(model):
            self._apply_outcome_to_belief(outcome, w)
        self._estimate_malleability()


# ─────────────────────────────────────────────
# Collective agent: social logit integration + social belief update
# ─────────────────────────────────────────────
class CollectiveMindsetAgent(MindsetAgent):
    """
    Overrides:
      act    → _collect_logits: self + neighbours × conformity_weight
      update → _collect_outcomes: self + neighbours × social_learning_weight
    Everything else inherited.
    """

    def __init__(self, model, prior, social_learning_weight=0.1, conformity_weight=0.1):
        super().__init__(model, prior)
        self.social_learning_weight = social_learning_weight
        self.conformity_weight = conformity_weight

    def _get_neighbors(self, model):
        return model.grid.get_neighbors(
            self.pos, moore=True, radius=model.radius, include_center=False)

    # ── act override ──────────────────────────

    def _collect_logits(self, model):
        sources = [(self.action_logits.copy(), 1.0)]
        for neighbor in self._get_neighbors(model):
            # neighbours' action_logits are populated because plan() ran for everyone first
            sources.append((neighbor.action_logits.copy(), self.conformity_weight))
        return sources

    # ── update override ───────────────────────

    def _collect_outcomes(self, model):
        sources = [(self.outcome_history[-1], 1.0)]
        for neighbor in self._get_neighbors(model):
            if neighbor.outcome_history:
                sources.append((neighbor.outcome_history[-1], self.social_learning_weight))
        return sources