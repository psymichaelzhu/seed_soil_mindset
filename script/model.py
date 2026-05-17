from mesa import Model
from mesa.space import SingleGrid
from mesa.datacollection import DataCollector
from agents import CollectiveMindsetAgent
import numpy as np


class MindsetModel(Model):
    def __init__(self, width=30, height=30,
                 reward_slope=1, reward_intercept=4, time_horizon=10, malleability=0.5,
                 growth_proportion=0.5, growth_prior_mean=0.7,growth_prior_strength=10,
                 social_learning_weight=0.1, conformity_weight=0.1,
                 seed=42):
        if seed is not None:
            seed = int(seed)
        super().__init__(rng=seed)

        # skill / environment parameters
        self.growth_proportion = growth_proportion
        self.reward_slope = reward_slope
        self.reward_intercept = reward_intercept
        self.time_horizon = time_horizon
        self.malleability = malleability  # true p(success | cultivate)

        # social parameters
        self.radius = 1

        # time step counter
        self.time_step = 0

        # torus grid so every agent has the same number of neighbours
        self.grid = SingleGrid(width, height, torus=True)

        # build priors: growth ~ Beta(7, 3), fixed ~ Beta(3, 7) when mean=0.7
        growth_prior = (growth_prior_mean * growth_prior_strength, (1 - growth_prior_mean) * growth_prior_strength)
        fixed_prior  = growth_prior[::-1]

        all_positions = list(self.grid.coord_iter())
        n_total = len(all_positions)
        n_growth = round(n_total * growth_proportion)

        # build a shuffled mindset list: exactly n_growth 'growth', rest 'fixed'
        mindset_list = ['growth'] * n_growth + ['fixed'] * (n_total - n_growth)
        self.random.shuffle(mindset_list)

        for (_, pos), mindset in zip(all_positions, mindset_list):
            if mindset == 'growth':
                agent = CollectiveMindsetAgent(
                    self, growth_prior, social_learning_weight, conformity_weight)
                agent.mindset = 'growth'
            else:
                agent = CollectiveMindsetAgent(
                    self, fixed_prior, social_learning_weight, conformity_weight)
                agent.mindset = 'fixed'

            # pre-draw cultivation outcomes for the whole horizon
            agent.cultivation_outcomes = [
                self.random.random() < self.malleability
                for _ in range(self.time_horizon)
            ]
            self.grid.place_agent(agent, pos)

        self.datacollector = DataCollector(
            model_reporters={
                "cultivate_proportion": lambda m: (
                    sum(1 for a in m.agents
                        if a.choice_history and a.choice_history[-1] == 'cultivate')
                    / len(m.agents)
                ),
                "growth_belief_mean": lambda m: np.mean([
                    a.malleability_estimate for a in m.agents if a.mindset == 'growth'
                ]),
                "growth_belief_std": lambda m: np.std([
                    a.malleability_estimate for a in m.agents if a.mindset == 'growth'
                ]),
                "fixed_belief_mean": lambda m: np.mean([
                    a.malleability_estimate for a in m.agents if a.mindset == 'fixed'
                ]),
                "fixed_belief_std": lambda m: np.std([
                    a.malleability_estimate for a in m.agents if a.mindset == 'fixed'
                ]),
            }
        )

    def reward_function(self, skill_level):
        return self.reward_intercept + self.reward_slope * skill_level

    def step(self):
        # ① all agents privately estimate and write action_logits
        self.agents.shuffle_do("plan", model=self)
        # ② all agents read neighbours' logits simultaneously → choose action
        #    (action_logits are NOT cleared yet, so every agent sees full planned choices)
        self.agents.shuffle_do("select", model=self)
        # ③ clear logits now that every agent has committed to an action
        self.agents.do("reset_logits", model=self)
        # ④ all agents execute their chosen action and observe the outcome
        self.agents.shuffle_do("observe", model=self)
        # ⑤ all agents update belief from neighbours' outcomes
        self.agents.shuffle_do("update", model=self)

        self.datacollector.collect(self)
        self.time_step += 1
        self.running = self.time_step < self.time_horizon