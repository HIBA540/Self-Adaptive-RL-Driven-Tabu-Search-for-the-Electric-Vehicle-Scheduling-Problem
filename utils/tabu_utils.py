from utils.constants import MAX_CANDIDATES

from utils.evaluator import evaluate_expected


def move_key(
    neighbor,
    current
):

    changed = [

        ev

        for ev in neighbor

        if neighbor[ev] != current.get(
            ev,
            (0, 0, 0)
        )
    ]

    return tuple(sorted(changed)[:6])


def generate_candidates(
    current,
    current_obj,
    best_obj,
    tabu_list,
    operator_selector,
    instance,
    objective
):

    candidates = []

    for _ in range(MAX_CANDIDATES):

        neighbor, op_idx = operator_selector(
            current,
            instance
        )

        obj_value = evaluate_expected(
            neighbor,
            instance,
            objective
        )

        move = move_key(
            neighbor,
            current
        )

        if (
            move not in tabu_list
            or obj_value < best_obj - 0.01
        ):

            candidates.append(
                (
                    neighbor,
                    obj_value,
                    move,
                    op_idx,
                    current_obj - obj_value
                )
            )

    return candidates
