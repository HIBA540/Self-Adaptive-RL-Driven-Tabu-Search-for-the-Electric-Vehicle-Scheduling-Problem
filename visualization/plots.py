import math
import os

import matplotlib.pyplot as plt
import numpy as np

from benchmark.statistics import compute_gaps

from operators import build_operators


def plot_quality_time_tradeoff(
    df,
    out_dir="results"
):

    if df is None or df.empty:
        return None

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    dfg = compute_gaps(df)

    aggregation = (

        dfg.groupby(
            ["objective", "method"]
        )

        .agg(

            avg_gap_pct=("gap_pct", "mean"),

            avg_time_s=("time_s", "mean"),
        )

        .reset_index()
    )

    objectives = list(
        aggregation["objective"].unique()
    )

    fig, axes = plt.subplots(

        1,

        len(objectives),

        figsize=(6 * len(objectives), 5),

        squeeze=False
    )

    axes = axes.ravel()

    for ax, objective in zip(axes, objectives):

        subset = aggregation[
            aggregation["objective"] == objective
        ]

        for _, row in subset.iterrows():

            ax.scatter(
                row["avg_time_s"],
                row["avg_gap_pct"],
                s=80
            )

            ax.annotate(
                row["method"],
                (
                    row["avg_time_s"],
                    row["avg_gap_pct"]
                )
            )

        ax.set_title(
            objective.replace("_", " ").title()
        )

        ax.set_xlabel(
            "Average runtime (s)"
        )

        ax.set_ylabel(
            "Average gap to best known (%)"
        )

        ax.grid(True)

    fig.tight_layout()

    path = os.path.join(
        out_dir,
        "quality_time_tradeoff.png"
    )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return path