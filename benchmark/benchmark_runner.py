import os
import time

import pandas as pd

from algorithms.tabu_classic import (
    tabu_search_classic
)

from algorithms.tabu_ucb import (
    tabu_search_mab
)

from algorithms.tabu_qlearning import (
    tabu_search_qlearning
)

from algorithms.tabu_qaps import (
    tabu_search_qaps
)

from benchmark.initialization import (
    greedy_initial
)

from benchmark.instance_generator import (
    generate_instance
)

from benchmark.statistics import (
    compute_gaps
)

from utils.evaluator import (
    evaluate_expected
)

from visualization.plots import (
    plot_quality_time_tradeoff
)


BENCHMARK_GRID = {

    "N_VALUES": [400],

    "SCENARIOS": [
        "Loose",
        "Tight75"
    ],

    "CONFIGS": [
        "balanced"
    ],

    "SEED_INDICES": [
        1,
        2,
        3
    ],

    "NUM_SAA": 15,

    "OBJECTIVES": [

        "total_tardiness",

        "max_tardiness",

        "total_waiting"
    ],
}


def run_full_benchmark(
    grid=None,
    out_dir="results"
):

    grid = grid or BENCHMARK_GRID

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    methods = [

        (
            "TS-Classic",
            tabu_search_classic
        ),

        (
            "TS-MAB",
            tabu_search_mab
        ),

        (
            "TS-QLearning",
            tabu_search_qlearning
        ),

        (
            "TS-QAPS",
            tabu_search_qaps
        ),
    ]

    rows = []

    combinations = [

        (
            n,
            scenario,
            config,
            seed
        )

        for n in grid["N_VALUES"]

        for scenario in grid["SCENARIOS"]

        for config in grid["CONFIGS"]

        for seed in grid["SEED_INDICES"]
    ]

    for (
        n,
        scenario,
        config,
        seed
    ) in combinations:

        instance_id = (
            f"n{n}_{scenario}_{config}_s{seed}"
        )

        instance = generate_instance(
            n,
            scenario,
            config,
            seed,
            num_scenarios=grid["NUM_SAA"]
        )

        print(
            f"\nRunning instance {instance_id}"
        )

        for objective in grid["OBJECTIVES"]:

            initial_solution = greedy_initial(
                instance,
                objective
            )

            greedy_obj = evaluate_expected(
                initial_solution,
                instance,
                objective
            )

            for method_name, method in methods:

                start = time.time()

                _, final_obj = method(
                    initial_solution.copy(),
                    instance,
                    objective
                )

                elapsed = (
                    time.time() - start
                )

                print(
                    f"{method_name:20s}"
                    f" | {objective:20s}"
                    f" | obj={final_obj:.2f}"
                    f" | t={elapsed:.1f}s"
                )

                rows.append({

                    "instance_id": instance_id,

                    "n": n,

                    "scenario": scenario,

                    "config": config,

                    "seed": seed,

                    "objective": objective,

                    "method": method_name,

                    "value": final_obj,

                    "time_s": elapsed,

                    "greedy_obj": greedy_obj,
                })

    df = pd.DataFrame(rows)

    df = compute_gaps(df)

    csv_path = os.path.join(
        out_dir,
        "results.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(
        f"\nResults saved to {csv_path}"
    )

    plot_quality_time_tradeoff(
        df,
        out_dir=out_dir
    )

    return df