import math
import random

from collections import defaultdict
from collections import deque

from utils.constants import NO_IMPROVE_LIMIT


class QLearningAOS:

    def __init__(
        self,
        operators,
        alpha=0.15,
        gamma=0.9,
        epsilon=0.3,
        n_instance=100
    ):

        self.operators = operators

        self.n_ops = len(operators)

        self.alpha = alpha

        self.gamma = gamma

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

        self.operator_history = deque(maxlen=5)

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
        reward
    ):

        q = self.q_table[state][action]

        q_next = max(
            self.q_table[new_state]
        )

        self.q_table[state][action] = (

            q

            + self.alpha * (
                reward
                + self.gamma * q_next
                - q
            )
        )

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )