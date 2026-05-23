import random


def op_swap_random(
    assignment,
    instance
):

    evs = list(assignment)

    if len(evs) < 2:
        return assignment.copy()

    ev1, ev2 = random.sample(evs, 2)

    p1, r1, _ = assignment[ev1]

    p2, r2, _ = assignment[ev2]

    neighbor = assignment.copy()

    neighbor[ev1] = (
        p2,
        r2,
        instance["evs"][ev1 - 1]["a_mean"]
    )

    neighbor[ev2] = (
        p1,
        r1,
        instance["evs"][ev2 - 1]["a_mean"]
    )

    return neighbor