from collections import defaultdict

import numpy as np

from utils.constants import TAU


def get_duration(energy, power):

    return int(np.ceil(energy / (power * TAU)))


def eval_scenario(
    assignment,
    instance,
    scenario_idx
):

    schedule = defaultdict(list)

    for ev_id, (power, charger, planned_start) in assignment.items():

        ev = instance["evs"][ev_id - 1]

        real_arrival = ev["a_scenarios"][scenario_idx]

        schedule[(power, charger)].append(
            (
                ev_id,
                real_arrival,
                planned_start,
                ev
            )
        )

    for key in schedule:
        schedule[key].sort(key=lambda x: x[2])

    total_tardiness = 0.0
    max_tardiness = 0.0
    total_waiting = 0.0

    for (power, charger), vehicles in schedule.items():

        current_time = 0

        for _, arrival, planned_start, ev in vehicles:

            start = max(
                arrival,
                planned_start,
                current_time
            )

            completion = (
                start
                + get_duration(ev["e"], power)
            )

            tardiness = max(
                0,
                completion - ev["d"]
            )

            total_tardiness += tardiness

            max_tardiness = max(
                max_tardiness,
                tardiness
            )

            total_waiting += max(
                0,
                start - arrival
            )

            current_time = completion

    return {
        "total_tardiness": total_tardiness,
        "max_tardiness": max_tardiness,
        "total_waiting": total_waiting,
    }


def evaluate_expected(
    assignment,
    instance,
    objective="total_tardiness"
):

    if objective == "max_tardiness":

        return float(np.mean([

            eval_scenario(
                assignment,
                instance,
                scenario_idx
            )["max_tardiness"]

            for scenario_idx in range(
                instance["num_scenarios"]
            )

        ]))

    return sum(

        instance["scenario_probabilities"][scenario_idx]

        * eval_scenario(
            assignment,
            instance,
            scenario_idx
        )[objective]

        for scenario_idx in range(
            instance["num_scenarios"]
        )
    )