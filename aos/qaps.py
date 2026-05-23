import math
import random

import numpy as np

from collections import defaultdict
from collections import deque

from utils.constants import NO_IMPROVE_LIMIT


class QLearningAOS_APS:

    ALPHA_LO = 0.05
    ALPHA_HI = 0.50

    GAMMA_LO = 0.70
    GAMMA_HI = 0.99

    SIGMA_ALPHA = 0.08
    SIGMA_GAMMA = 0.08

    P_EXPLORE = 0.40

    Q_TABLE_MATURITY = 20

    def __init__(
        self,
        operators,
        base_alpha=0.15,
        base_gamma=0.90,
        epsilon=0.3,
        n_instance=100
    ):

        self.operators = operators

        self.n_ops = len(operators)

        self.base_alpha = base_alpha

        self.base_gamma = base_gamma

        self.epsilon = epsilon

        self.epsilon_min = 0.10

        target_iters = max(
            500,
            2 * NO_IMPROVE_LIMIT
            * int(math.log(n_instance + 1) + 1)
        )

        if epsilon > self.epsilon_min:

            self.epsilon_decay = (
                self.epsilon_min / epsilon
            ) ** (1.0 / target_iters)

        else:
            self.epsilon_decay = 1.0

        self.q_table = defaultdict(
            lambda: [0.0] * self.n_ops
        )

        self.best_alpha = base_alpha

        self.best_gamma = base_gamma

        self.best_perf = float("inf")

        self.operator_history = deque(maxlen=5)

    def _sample_hyperparams(self):

        if (

            random.random() < self.P_EXPLORE

            or len(self.q_table)
            < self.Q_TABLE_MATURITY
        ):

            alpha = random.uniform(
                self.ALPHA_LO,
                self.ALPHA_HI
            )

            gamma = random.uniform(
                self.GAMMA_LO,
                self.GAMMA_HI
            )

        else:

            alpha = float(np.clip(

                np.random.normal(
                    self.best_alpha,
                    self.SIGMA_ALPHA
                ),

                self.ALPHA_LO,
                self.ALPHA_HI
            ))

            gamma = float(np.clip(

                np.random.normal(
                    self.best_gamma,
                    self.SIGMA_GAMMA
                ),

                self.GAMMA_LO,
                self.GAMMA_HI
            ))

        return alpha, gamma

    def select(
        self,
        state
    ):

        if random.random() < self.epsilon:

            action = random.randint(
                0,
                self.n_ops - 1
            )

        else:

            q_values = self.q_table[state]

            max_q = max(q_values)

            action = random.choice([

                i

                for i, q in enumerate(q_values)

                if q == max_q
            ])

        self.operator_history.append(action)

        return action

    def get_last_operators(self):

        return list(self.operator_history)

    def update_explicit(
        self,
        state,
        action,
        new_state,
        reward,
        current_obj
    ):

        q = self.q_table[state][action]

        q_next = max(
            self.q_table[new_state]
        )

        q_base = (

            q

            + self.base_alpha * (
                reward
                + self.base_gamma * q_next
                - q
            )
        )

        alpha_sampled, gamma_sampled = (
            self._sample_hyperparams()
        )

        q_adapt = (

            q

            + alpha_sampled * (
                reward
                + gamma_sampled * q_next
                - q
            )
        )

        self.q_table[state][action] = max(
            q_base,
            q_adapt
        )

        if (
            q_adapt > q_base + 1e-6
            and current_obj < self.best_perf
        ):

            self.best_perf = current_obj

            self.best_alpha = alpha_sampled

            self.best_gamma = gamma_sampled

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )