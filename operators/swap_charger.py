import random


def op_swap_charger(
    assignment,
    instance
):

    ev_id = random.choice(
        list(assignment)
    )

    power, old_charger, start = assignment[ev_id]

    candidates = [

        charger

        for charger in instance["charger_ids"][power]

        if charger != old_charger
    ]

    if not candidates:
        return assignment.copy()

    neighbor = assignment.copy()

    neighbor[ev_id] = (
        power,
        random.choice(candidates),
        start
    )

    return neighbor