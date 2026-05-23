import pandas as pd

from scipy.stats import wilcoxon


def compute_gaps(df):

    df = df.copy()

    df["best_known"] = (

        df.groupby(
            ["instance_id", "objective"]
        )["value"]

        .transform("min")
    )

    df["gap_pct"] = (

        100.0

        * (
            df["value"]
            - df["best_known"]
        )

        / df["best_known"].clip(lower=1e-9)
    )

    return df


def summary_table(
    df,
    groupby=None
):

    df = compute_gaps(df)

    df["is_best"] = (
        df["gap_pct"] < 0.01
    ).astype(int)

    keys = ["method"]

    if groupby:

        keys = (

            keys + [groupby]

            if isinstance(groupby, str)

            else keys + list(groupby)
        )

    aggregation = (

        df.groupby(keys)

        .agg(

            avg_gap=("gap_pct", "mean"),

            std_gap=("gap_pct", "std"),

            n_best=("is_best", "sum"),

            avg_time=("time_s", "mean"),
        )

        .reset_index()
    )

    print("\n" + "=" * 72)

    label = (

        f"Summary — groupby={groupby}"

        if groupby

        else "Summary — GLOBAL"
    )

    print(f" {label} ".center(72))

    print("=" * 72)

    print(

        aggregation.to_string(

            index=False,

            float_format="{:.3f}".format
        )
    )

    return aggregation


def wilcoxon_table(
    df,
    label=""
):

    df = compute_gaps(df)

    methods = sorted(
        df["method"].unique()
    )

    print("\n" + "=" * 72)

    print(
        f" Wilcoxon Signed-Rank Tests "
        f"(p-values) — {label} ".center(72)
    )

    print("=" * 72)

    header = (

        f"{'':22s}"

        + "".join(
            f"{m:>16s}"
            for m in methods
        )
    )

    print(header)

    print("-" * len(header))

    for method_A in methods:

        row = f"{method_A:22s}"

        gaps_A = (

            df[df["method"] == method_A]

            .sort_values(
                ["instance_id", "objective"]
            )["gap_pct"]

            .values
        )

        for method_B in methods:

            if method_A == method_B:

                row += f"{'—':>16s}"

            else:

                gaps_B = (

                    df[df["method"] == method_B]

                    .sort_values(
                        ["instance_id", "objective"]
                    )["gap_pct"]

                    .values
                )

                n = min(
                    len(gaps_A),
                    len(gaps_B)
                )

                try:

                    _, p_value = wilcoxon(
                        gaps_A[:n],
                        gaps_B[:n]
                    )

                    significance = (

                        "**"

                        if p_value < 0.05

                        else (
                            "*"
                            if p_value < 0.10
                            else ""
                        )
                    )

                    row += (
                        f"{p_value:>13.2e}"
                        f"{significance:>3s}"
                    )

                except Exception:

                    row += f"{'N/A':>16s}"

        print(row)