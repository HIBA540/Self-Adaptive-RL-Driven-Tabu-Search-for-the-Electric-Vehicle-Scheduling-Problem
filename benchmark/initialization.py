from utils.evaluator import get_duration


def greedy_initial(
    instance,
    objective="total_tardiness"
):

    evs = instance["evs"].copy()

    if objective in (
        "total_tardiness",
        "max_tardiness"
    ):

        evs = sorted(

            evs,

            key=lambda x: (
                x["d"],
                x["a_mean"],
                -x["e"]
            )
        )

    else:

        evs = sorted(

            evs,

            key=lambda x: (
                x["a_mean"],
                x["d"]
            )
        )

    end_times = {

        power: [0] * count

        for power, count in zip(
            instance["powers"],
            instance["counts"]
        )
    }

    assignment = {}

    for ev in evs:

        best_power = None
        best_charger = None
        best_start = None

        best_value = float("inf")

        for power in instance["powers"]:

            for charger in instance["charger_ids"][power]:

                start = max(
                    ev["a_mean"],
                    end_times[power][charger]
                )

                completion = (

                    start
                    + get_duration(
                        ev["e"],
                        power
                    )
                )

                value = (

                    start - ev["a_mean"]

                    if objective == "total_waiting"

                    else max(
                        0,
                        completion - ev["d"]
                    )
                )

                if value < best_value:

                    best_value = value

                    best_power = power

                    best_charger = charger

                    best_start = start

        if best_power is not None:

            assignment[ev["id"]] = (

                best_power,
                best_charger,
                best_start
            )

            end_times[best_power][best_charger] = (

                best_start
                + get_duration(
                    ev["e"],
                    best_power
                )
            )

    return assignment