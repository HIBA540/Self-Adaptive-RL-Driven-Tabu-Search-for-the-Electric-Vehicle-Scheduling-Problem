import math


class UCB_AOS:

    def __init__(
        self,
        operators,
        C=1.0
    ):

        self.operators = operators

        self.n_ops = len(operators)

        self.C = C

        self.counts = [0] * self.n_ops

        self.values = [0.0] * self.n_ops

        self.total = 0

    def select(self):

        self.total += 1

        for i in range(self.n_ops):

            if self.counts[i] == 0:
                return i

        scores = [

            self.values[i] / self.counts[i]

            + self.C * math.sqrt(
                math.log(self.total)
                / self.counts[i]
            )

            for i in range(self.n_ops)
        ]

        return scores.index(max(scores))

    def update(
        self,
        idx,
        reward
    ):

        self.counts[idx] += 1

        self.values[idx] += reward