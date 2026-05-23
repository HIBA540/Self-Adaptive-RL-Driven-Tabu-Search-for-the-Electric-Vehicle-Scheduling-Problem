from collections import deque

import random
import time
import numpy as np

from aos.qlearning import QLearningAOS

from operators import build_operators

from utils.constants import (
    TABU_LIST_LENGTH,
    TIME_LIMIT,
    NO_IMPROVE_LIMIT,
    NO_IMPROVE_EMPTY_PENALTY,
)

from utils.evaluator import evaluate_expected

from utils.state_representation import extract_state

from utils.tabu_utils import (
    move_key,
)

from utils.constants import MAX_CANDIDATES


def tabu_search_qlearning(
    initial_assignment,
    instance,
    objective="total_tardiness",
    **kwargs
):

    n_instance = len(instance["evs"])

    operators = build_operators(objective)

    operator_functions = [
        func for _, func in operators
    ]

    qlearning = QLearningAOS(
        operator_functions,
        n_instance=n_instance
    )

    best = initial_assignment.copy()

    best_obj = evaluate_expected(
        best,
        instance,
        objective
    )

    current = best.copy()

    current_obj = best_obj

    tabu = deque(
        maxlen=TABU_LIST_LENGTH
    )

    start = time.time()

    no_improve = 0

    iteration = 0

    last_improvement = 0.0

    while (

        time.time() - start < TIME_LIMIT

        and no_improve < NO_IMPROVE_LIMIT
    ):

        iteration += 1

        candidates = []

        last_operators = (
            qlearning.get_last_operators()
        )

        state = extract_state(
            current,
            instance,
            iteration=iteration,
            max_iter=NO_IMPROVE_LIMIT,
            last_improvement=last_improvement,
            best_obj=best_obj,
            curr_obj=current_obj,
            last_operators=last_operators
        )

        for _ in range(MAX_CANDIDATES):

            op_idx = qlearning.select(state)

            neighbor = operator_functions[op_idx](
                current,
                instance
            )

            neighbor_obj = evaluate_expected(
                neighbor,
                instance,
                objective
            )

            move = move_key(
                neighbor,
                current
            )

            if (
                move not in tabu
                or neighbor_obj < best_obj - 0.01
            ):

                candidates.append(
                    (
                        neighbor,
                        neighbor_obj,
                        move,
                        op_idx
                    )
                )

        if not candidates:

            qlearning.update_explicit(
                state,
                random.randint(
                    0,
                    len(operator_functions) - 1
                ),
                state,
                0.0
            )

            no_improve += (
                NO_IMPROVE_EMPTY_PENALTY
            )

            continue

        mean_obj = np.mean([
            c[1] for c in candidates
        ])

        candidates.sort(
            key=lambda x: x[1]
        )

        (
            neighbor,
            neighbor_obj,
            move,
            op_idx
        ) = candidates[0]

        centered_reward = (
            mean_obj - neighbor_obj
        )

        new_state = extract_state(
            neighbor,
            instance,
            iteration=iteration,
            max_iter=NO_IMPROVE_LIMIT,
            last_improvement=centered_reward,
            best_obj=best_obj,
            curr_obj=neighbor_obj,
            last_operators=qlearning.get_last_operators()
        )

        qlearning.update_explicit(
            state,
            op_idx,
            new_state,
            centered_reward
        )

        last_improvement = (
            centered_reward
        )

        current = neighbor

        current_obj = neighbor_obj

        tabu.append(move)

        if neighbor_obj < best_obj - 0.01:

            best = current.copy()

            best_obj = neighbor_obj

            no_improve = 0

        else:
            no_improve += 1

    return best, best_obj