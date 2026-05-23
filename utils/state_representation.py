import numpy as np

from utils.constants import NO_IMPROVE_LIMIT

from utils.evaluator import get_duration


def extract_state(
    assignment,
    instance,
    iteration=0,
    max_iter=NO_IMPROVE_LIMIT,
    last_improvement=0.0,
    best_obj=None,
    curr_obj=None,
    last_operators=None
):

    delays = []

    loads = []

    n = len(instance["evs"])

    for ev_id, (power, charger, start) in assignment.items():

        ev = instance["evs"][ev_id - 1]

        effective_start = max(
            start,
            ev["a_mean"]
        )

        completion = (
            effective_start
            + get_duration(ev["e"], power)
        )

        delays.append(
            max(
                0.0,
                completion - ev["d"]
            )
        )

        loads.append(power)

    if n == 0:

        return (
            0,
            0,
            0,
            0,
            0,
            0
        )

    late_ratio = sum(
        1 for delay in delays if delay > 0
    ) / n

    f1 = min(
        4,
        int(late_ratio * 5)
    )

    f2 = min(
        3,
        int(4 * iteration / max(max_iter, 1))
    )

    avg_delay = (
        float(np.mean(delays))
        if delays else 0.0
    )

    max_delay = (
        float(np.max(delays))
        if delays else 0.0
    )

    concentration = (
        max_delay / (avg_delay + 1e-6)
    )

    f3 = min(
        3,
        int(concentration / 2)
    )

    load_balance = (

        float(np.std(loads))
        / (np.mean(loads) + 1e-6)

        if loads else 0.0
    )

    f4 = min(
        2,
        int(5 * load_balance)
    )

    if (
        best_obj
        and best_obj > 1e-9
        and curr_obj
    ):

        gap = (
            curr_obj - best_obj
        ) / best_obj

        f5 = min(
            3,
            int(gap * 10)
        )

    else:
        f5 = 0

    if (
        last_operators
        and len(last_operators) > 0
    ):

        last_op = last_operators[-1]

        f6 = min(
            5,
            last_op + 1
        )

    else:
        f6 = 0

    return (
        f1,
        f2,
        f3,
        f4,
        f5,
        f6
    )