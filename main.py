from benchmark.benchmark_runner import (
    run_full_benchmark
)

from benchmark.statistics import (
    summary_table,
    wilcoxon_table,
)


if __name__ == "__main__":

    df = run_full_benchmark()

    summary_table(df)

    summary_table(
        df,
        groupby="scenario"
    )

    summary_table(
        df,
        groupby="n"
    )

    wilcoxon_table(
        df,
        label="All Instances"
    )

    print("\nBenchmark completed.")