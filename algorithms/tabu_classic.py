from collections import deque

import random
import time

from operators import build_operators

from utils.constants import (
    TABU_LIST_LENGTH,
    TIME_LIMIT,
    NO_IMPROVE_LIMIT,
    NO_IMPROVE_EMPTY_PENALTY,
)

from utils.evaluator import evaluate_expected

from utils.tabu_utils import generate_candidates


def tabu_search_classic(
    initial_assignment,
    instance,
    objective="total_tardiness",
    **kwargs
):

    operators = build_operators(objective)

    operator_names = [
        name for name, _ in operators
    ]

    operator_functions = [
        func for _, func in operators
    ]

    n_operators = len(operator_functions)

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

    def operator_selector(
        current_solution,
        current_instance
    ):

        idx = random.randint(
            0,
            n_operators - 1
        )

        return (
            operator_functions[idx](
                current_solution,
                current_instance
            ),
            idx
        )

    while (

        time.time() - start < TIME_LIMIT

        and no_improve < NO_IMPROVE_LIMIT
    ):

        iteration += 1

        candidates = generate_candidates(

            current,
            current_obj,
            best_obj,
            tabu,
            operator_selector,
            instance,
            objective
        )

        if not candidates:

            no_improve += (
                NO_IMPROVE_EMPTY_PENALTY
            )

            continue

        candidates.sort(
            key=lambda x: x[1]
        )

        (
            neighbor,
            neighbor_obj,
            move,
            op_idx,
            reward
        ) = candidates[0]

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