#!/usr/bin/env python3
"""Query latency visualiser — run with: python plot_latency.py"""

import subprocess, sys

def _ensure(pkg):
    try:
        __import__(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("matplotlib")

import csv
import io
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA = """builder,n_tokens,n_threads,t_total_s,t_sa_s,mem_mb,lat_1g_med,lat_1g_p99,lat_2g_med,lat_2g_p99,lat_4g_med,lat_4g_p99
caps_sa,5010000,20,5.077,4.175,391.3,54.38,126.27,69.53,106.98,80.72,106.83
caps_sa,25050000,20,17.814,13.215,1854.6,60.28,166.53,77.83,111.12,94.76,122.35
caps_sa,50100000,20,33.142,24.03,2581.2,63.99,232.74,82.6,121.8,97.08,135.26
caps_sa,250500000,20,166.848,118.357,18925.7,68.27,227.44,90.73,192.19,110.37,168.58
caps_sa,501000000,20,382.293,284.356,38994.1,71.88,286.46,96.24,242.49,117.02,214.63
rust,5010000,20,1.968,1.034,0.0,61.57,222.17,78.91,119.13,91.48,144.4
rust,25050000,20,12.782,8.039,0.0,68.71,256.02,88.01,144.83,107.98,147.52
rust,50100000,20,26.2,16.805,0.0,71.46,276.75,92.47,162.44,108.17,157.49
rust,250500000,20,137.62,87.511,0.0,77.5,278.13,101.92,218.93,124.08,199.99
rust,501000000,20,272.556,172.923,0.0,81.37,301.89,107.93,243.7,131.45,218.31"""

LAT_COLS = ["lat_1g_med", "lat_1g_p99", "lat_2g_med", "lat_2g_p99", "lat_4g_med", "lat_4g_p99"]

GROUPS = [
    ("1G", ["lat_1g_med", "lat_1g_p99"]),
    ("2G", ["lat_2g_med", "lat_2g_p99"]),
    ("4G", ["lat_4g_med", "lat_4g_p99"]),
]

BUILDER_COLOR = {"caps_sa": "#4C72B0", "rust": "#DD8452"}


def parse_data():
    rows = list(csv.DictReader(io.StringIO(DATA)))
    builders = sorted({r["builder"] for r in rows})
    data = {}
    for b in builders:
        subset = sorted([r for r in rows if r["builder"] == b], key=lambda r: int(r["n_tokens"]))
        data[b] = {
            "n_tokens": [int(r["n_tokens"]) for r in subset],
            **{col: [float(r[col]) for r in subset] for col in LAT_COLS},
        }
    return data, builders


def plot(data, builders):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Query Latency by Builder & Index Size", fontsize=15, fontweight="bold", y=1.01)

    tick_vals = data[builders[0]]["n_tokens"]
    tick_labels = [
        f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}K"
        for v in tick_vals
    ]

    for ax, (gram_label, cols) in zip(axes, GROUPS):
        for b in builders:
            xs = data[b]["n_tokens"]
            color = BUILDER_COLOR[b]
            for col in cols:
                is_p99 = col.endswith("_p99")
                ls, marker = ("--", "s") if is_p99 else ("-", "o")
                ax.plot(
                    xs, data[b][col],
                    linestyle=ls, marker=marker, color=color,
                    linewidth=2, markersize=5, alpha=0.85,
                    label=f"{b}  {'P99' if is_p99 else 'Median'}",
                )

        ax.set_title(f"{gram_label} Queries", fontweight="bold")
        ax.set_xlabel("Corpus tokens")
        ax.set_ylabel("Latency (ms)")
        ax.set_xscale("log")

        # Pin one tick per actual corpus index value
        ax.set_xticks(tick_vals)
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.xaxis.set_minor_locator(mticker.NullLocator())  # suppress minor ticks

        ax.grid(True, which="major", linestyle=":", linewidth=0.6, alpha=0.7)

        handles, labels = ax.get_legend_handles_labels()
        seen = dict(zip(labels, handles))  # deduplicate
        ax.legend(seen.values(), seen.keys(), fontsize=8, loc="upper left")

    plt.tight_layout()
    out = "query_latency.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    data, builders = parse_data()
    plot(data, builders)