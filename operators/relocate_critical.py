import random

from utils.evaluator import get_duration


def op_relocate_critical(
    assignment,
    instance,
    objective="total_tardiness"
):

    critical = [

        ev_id

        for ev_id in assignment

        if (

            max(
                assignment[ev_id][2],
                instance["evs"][ev_id - 1]["a_mean"]
            )

            + get_duration(
                instance["evs"][ev_id - 1]["e"],
                assignment[ev_id][0]
            )

            > instance["evs"][ev_id - 1]["d"]
        )
    ]

    ev_id = random.choice(
        critical or list(assignment)
    )

    max_power = max(
        instance["powers"]
    )

    charger = random.choice(
        instance["charger_ids"][max_power]
    )

    neighbor = assignment.copy()

    neighbor[ev_id] = (
        max_power,
        charger,
        instance["evs"][ev_id - 1]["a_mean"]
    )

    return neighbor