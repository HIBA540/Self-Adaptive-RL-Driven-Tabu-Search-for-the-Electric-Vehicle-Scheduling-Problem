from collections import deque

import time

from aos.ucb import UCB_AOS

from operators import build_operators

from utils.constants import (
    TABU_LIST_LENGTH,
    TIME_LIMIT,
    NO_IMPROVE_LIMIT,
    NO_IMPROVE_EMPTY_PENALTY,
)

from utils.evaluator import evaluate_expected

from utils.tabu_utils import generate_candidates


def tabu_search_mab(
    initial_assignment,
    instance,
    objective="total_tardiness",
    ucb_C=1.0,
    **kwargs
):

    operators = build_operators(objective)

    operator_functions = [
        func for _, func in operators
    ]

    aos = UCB_AOS(
        operator_functions,
        C=ucb_C
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

    def operator_selector(
        current_solution,
        current_instance
    ):

        idx = aos.select()

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
            improvement
        ) = candidates[0]

        aos.update(
            op_idx,
            improvement
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