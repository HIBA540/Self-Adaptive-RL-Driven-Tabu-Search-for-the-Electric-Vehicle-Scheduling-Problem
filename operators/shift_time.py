import random


def op_shift_time(
    assignment,
    instance
):

    ev_id = random.choice(
        list(assignment)
    )

    power, charger, start = assignment[ev_id]

    ev = instance["evs"][ev_id - 1]

    new_start = max(
        ev["a_mean"],
        start + random.randint(-3, 3)
    )

    neighbor = assignment.copy()

    neighbor[ev_id] = (
        power,
        charger,
        new_start
    )

    return neighbor