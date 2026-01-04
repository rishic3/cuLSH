"""Plot benchmark results from JSON reports."""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np


def load_report(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_comparison(gpu_report: dict, cpu_report: dict, output: str | None = None):
    """Create bar chart comparing GPU vs CPU performance."""

    # Extract times
    gpu_fit = gpu_report["runtimes"]["fit_time"]
    gpu_query = gpu_report["runtimes"]["query_time"]
    cpu_fit = cpu_report["runtimes"]["fit_time"]
    cpu_query = cpu_report["runtimes"]["query_time"]

    # Calculate throughput (queries/sec)
    n_queries = gpu_report["params"]["n_queries"]
    gpu_throughput = n_queries / gpu_query
    cpu_throughput = n_queries / cpu_query

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Colors
    gpu_color = "#76b900"  # NVIDIA green
    cpu_color = "#0071c5"  # Intel blue

    # Plot 1: Runtime comparison (in milliseconds, log scale)
    ax1 = axes[0]
    x = np.arange(2)
    width = 0.35

    # Convert to milliseconds
    gpu_times = [gpu_fit * 1000, gpu_query * 1000]
    cpu_times = [cpu_fit * 1000, cpu_query * 1000]

    bars1 = ax1.bar(
        x - width / 2, gpu_times, width, label="cuLSH (GPU)", color=gpu_color
    )
    bars2 = ax1.bar(
        x + width / 2, cpu_times, width, label="FAISS (CPU)", color=cpu_color
    )

    ax1.set_yscale("log")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Runtime Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["Index Build", "Query"])
    ax1.legend()

    # Add value labels on bars (in ms)
    for bar in bars1:
        val = bar.get_height()
        ax1.annotate(
            f"{val:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, val * 1.1),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    for bar in bars2:
        val = bar.get_height()
        ax1.annotate(
            f"{val:.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, val * 1.1),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add headroom for annotations
    ax1.set_ylim(top=ax1.get_ylim()[1] * 2)

    # Plot 2: Throughput comparison (log scale)
    ax2 = axes[1]
    throughputs = [gpu_throughput, cpu_throughput]
    colors = [gpu_color, cpu_color]
    labels = ["cuLSH\n(GPU)", "FAISS\n(CPU)"]

    bars = ax2.bar(labels, throughputs, color=colors)
    ax2.set_yscale("log")
    ax2.set_ylabel("Queries / second")

    # Add value labels on bars
    for bar, val in zip(bars, throughputs):
        ax2.annotate(
            f"{val:,.0f}",
            xy=(bar.get_x() + bar.get_width() / 2, val * 1.1),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Add headroom for annotations
    ax2.set_ylim(top=ax2.get_ylim()[1] * 2)
    ax2.set_title("Query Throughput")

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved to {output}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark comparison")
    parser.add_argument("--gpu", required=True, help="Path to GPU report JSON")
    parser.add_argument("--cpu", required=True, help="Path to CPU report JSON")
    parser.add_argument(
        "-o", "--output", help="Output image path (shows plot if not specified)"
    )
    args = parser.parse_args()

    gpu_report = load_report(args.gpu)
    cpu_report = load_report(args.cpu)

    plot_comparison(gpu_report, cpu_report, args.output)


if __name__ == "__main__":
    main()
