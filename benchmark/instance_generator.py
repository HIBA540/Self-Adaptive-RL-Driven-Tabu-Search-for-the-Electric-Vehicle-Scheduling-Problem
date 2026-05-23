import hashlib

import numpy as np

from utils.constants import (
    TAU,
    MAX_POWER,
    T_HORIZON_FACTOR,
    CHARGER_POWERS,
    CHARGER_CONFIGS,
)


def set_seed(seed_string):

    return int(

        hashlib.md5(
            seed_string.encode()
        ).hexdigest(),

        16

    ) % (2**32)


def bimodal_arrivals(
    n,
    rng
):

    choice = rng.choice(
        [0, 1],
        size=n,
        p=[0.6, 0.4]
    )

    arrivals = np.where(

        choice == 0,

        rng.normal(32, 6, n),

        rng.normal(72, 6, n)
    )

    return np.clip(

        np.ceil(arrivals / TAU),

        1,

        None

    ).astype(int)


def stochastic_arrivals(
    n,
    rng,
    num_scenarios,
    scenario_type
):

    scenarios = []

    if scenario_type == "PeakTight80":

        for _ in range(num_scenarios):

            base = bimodal_arrivals(
                n,
                rng
            )

            eps = rng.uniform(
                -0.2,
                0.2,
                n
            )

            scenarios.append(

                np.clip(

                    np.ceil(base * (1 + eps)),

                    1,

                    None

                ).astype(int)
            )

    else:

        for _ in range(num_scenarios):

            base = rng.integers(

                1,

                int(0.2 * n / TAU) + 1,

                size=n
            )

            eps = rng.uniform(
                -0.3,
                0.3,
                n
            )

            scenarios.append(

                np.clip(

                    np.ceil(base * (1 + eps)),

                    1,

                    None

                ).astype(int)
            )

    return (
        scenarios,
        [1.0 / num_scenarios] * num_scenarios
    )


def bimodal_energy(
    n,
    rng
):

    choice = rng.choice(
        [0, 1],
        size=n,
        p=[0.7, 0.3]
    )

    energy = np.where(

        choice == 0,

        rng.normal(30, 8, n),

        rng.normal(60, 10, n)
    )

    return np.clip(
        energy,
        5.5,
        66
    )


def generate_instance(
    n,
    scenario,
    config,
    seed_idx,
    num_scenarios=15
):

    seed_string = (

        f"stochastic_n{n}_"
        f"{scenario}_{config}_"
        f"seed_{seed_idx:02d}"
    )

    rng = np.random.default_rng(
        set_seed(seed_string)
    )

    arrival_scenarios, probabilities = (
        stochastic_arrivals(
            n,
            rng,
            num_scenarios,
            scenario
        )
    )

    mean_arrivals = np.mean(
        arrival_scenarios,
        axis=0
    ).astype(int)

    energy = bimodal_energy(
        n,
        rng
    )

    best_processing = np.ceil(
        energy / (MAX_POWER * TAU)
    ).astype(int)

    if scenario == "Loose":

        alpha = np.clip(

            1.2
            - 0.15 * (
                energy
                / (MAX_POWER * TAU)
            ),

            0.5,
            1.0
        )

    else:

        tight_ratio = (
            0.80
            if scenario == "PeakTight80"
            else 0.75
        )

        is_tight = (
            rng.random(n) < tight_ratio
        )

        alpha = np.where(
            is_tight,
            0.1,
            0.2
        )

    slack = np.ceil(
        alpha * best_processing
    ).astype(int)

    deadlines = np.minimum(

        mean_arrivals
        + best_processing
        + slack,

        int(T_HORIZON_FACTOR * n / TAU)
    )

    counts = CHARGER_CONFIGS[config]

    evs = [

        {
            "id": i + 1,

            "a_scenarios": [

                int(a[i])

                for a in arrival_scenarios
            ],

            "a_mean": int(
                mean_arrivals[i]
            ),

            "d": int(
                deadlines[i]
            ),

            "e": float(
                energy[i]
            ),
        }

        for i in range(n)
    ]

    return {

        "evs": evs,

        "powers": CHARGER_POWERS,

        "counts": counts,

        "charger_ids": {

            p: list(range(c))

            for p, c in zip(
                CHARGER_POWERS,
                counts
            )
        },

        "seed_str": seed_string,

        "scenario_probabilities": probabilities,

        "num_scenarios": num_scenarios,

        "meta": {

            "n": n,

            "scenario": scenario,

            "config": config,

            "seed_idx": seed_idx,
        },
    }