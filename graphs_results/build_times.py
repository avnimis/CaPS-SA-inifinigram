#!/usr/bin/env python3
"""Build time visualiser — run with: python plot_build_times.py"""

import csv
import io
import matplotlib.pyplot as plt
import numpy as np

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

COLORS = {
    "caps_sa": {"total": "#4C72B0", "sa": "#9ab5da"},
    "rust":    {"total": "#DD8452", "sa": "#edb899"},
}


def parse_data():
    rows = list(csv.DictReader(io.StringIO(DATA)))
    builders = sorted({r["builder"] for r in rows})
    data = {}
    for b in builders:
        subset = sorted([r for r in rows if r["builder"] == b], key=lambda r: int(r["n_tokens"]))
        data[b] = {
            "n_tokens": [int(r["n_tokens"]) for r in subset],
            "t_total":  [float(r["t_total_s"]) for r in subset],
            "t_sa":     [float(r["t_sa_s"]) for r in subset],
            "t_other":  [float(r["t_total_s"]) - float(r["t_sa_s"]) for r in subset],
        }
    return data, builders


def plot(data, builders):
    n_tokens  = data[builders[0]]["n_tokens"]
    x_labels  = [f"{v/1e6:.0f}M" if v >= 1e6 else f"{v/1e3:.0f}K" for v in n_tokens]
    n_groups  = len(n_tokens)
    n_bars    = len(builders) * 2          # total + SA-only per builder
    bar_w     = 0.15
    group_gap = bar_w * (n_bars + 1)
    x         = np.arange(n_groups) * group_gap

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Build times by builder & corpus size", fontsize=14, fontweight="bold")

    offsets = np.linspace(-(n_bars - 1) / 2, (n_bars - 1) / 2, n_bars) * bar_w
    bar_idx = 0
    handles = []

    for b in builders:
        for kind, key in [("total", "t_total"), ("SA only", "t_sa")]:
            color = COLORS[b]["total" if kind == "total" else "sa"]
            bars  = ax.bar(
                x + offsets[bar_idx],
                data[b][key],
                width=bar_w,
                color=color,
                label=f"{b} — {kind}",
                zorder=3,
            )
            handles.append(bars)
            bar_idx += 1

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Corpus size (tokens)", fontsize=11)
    ax.set_ylabel("Time (s)", fontsize=11)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, ncol=2, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = "build_times.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")
    plt.show()


if __name__ == "__main__":
    data, builders = parse_data()
    plot(data, builders)