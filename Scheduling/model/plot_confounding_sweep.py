import argparse
import json
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

METHODS = ["GESPI", "CCKE", "Naive"]
TARGETS = {
    "Y_0": ("cov_0", "width_0", "Y0"),
    "Y_1": ("cov_1", "width_1", "Y1"),
    "ITE": ("cov_ite", "width_ite", "ITE"),
}
COLORS = {
    "GESPI": "#2f6fbb",
    "CCKE": "#c8524a",
    "Naive": "#5c8a8a",
}

def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _fmt_param(value):
    return str(value).replace(".", "p")

def _parse_conf_from_path(path):
    match = re.search(r"_conf([0-9]+(?:p[0-9]+)?)\.json$", os.path.basename(path))
    if match is None:
        return None
    return float(match.group(1).replace("p", "."))

def _load_results(results_dir, alpha, epsilon):
    runs = []
    for name in sorted(os.listdir(results_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(results_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)

        if not np.isclose(float(result.get("alpha", np.nan)), alpha):
            continue
        if not np.isclose(float(result.get("epsilon", np.nan)), epsilon):
            continue

        confounding_strength = result.get("confounding_strength")
        if confounding_strength is None:
            confounding_strength = _parse_conf_from_path(path)
        if confounding_strength is None:
            continue

        runs.append((float(confounding_strength), result))

    runs.sort(key=lambda item: item[0])
    if not runs:
        raise ValueError(
            f"No saved result files found in {results_dir!r} for alpha={alpha} and epsilon={epsilon}."
        )
    return runs

def _metric_values(result, method, metric):
    return np.asarray(
        [fold["methods"][method][metric] for fold in result["folds"]],
        dtype=float,
    )

def _draw_box_group(ax, grouped_values, confounding_strengths, ylabel, title, target_level=None):
    centers = np.arange(len(confounding_strengths), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(METHODS))
    box_width = 0.14
    flierprops = {
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 3.5,
        "alpha": 0.75,
    }

    for method_id, method in enumerate(METHODS):
        values = grouped_values[method]
        boxes = ax.boxplot(
            values,
            positions=centers + offsets[method_id],
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            flierprops=flierprops,
        )
        for patch in boxes["boxes"]:
            patch.set_facecolor(COLORS[method])
            patch.set_alpha(0.58)
            patch.set_edgecolor("black")
        for median in boxes["medians"]:
            median.set_color("black")
            median.set_linewidth(1.2)
        ax.plot([], [], color=COLORS[method], linewidth=8, alpha=0.58, label=method)

    if target_level is not None:
        ax.axhline(
            target_level,
            color="black",
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            label=f"1-alpha={target_level:.2f}",
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(centers)
    ax.set_xticklabels([f"{value:g}" for value in confounding_strengths])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.24))

def plot_confounding_sweep(
    runs,
    target,
    out_dir,
    alpha=0.2,
    epsilon=0.05,
    efficiency_scale=1e6,
):
    cov_metric, width_metric, target_label = TARGETS[target]
    confounding_strengths = [conf for conf, _ in runs]

    coverage_values = {method: [] for method in METHODS}
    efficiency_values = {method: [] for method in METHODS}
    for _, result in runs:
        for method in METHODS:
            coverage_values[method].append(_metric_values(result, method, cov_metric))
            efficiency_values[method].append(
                _metric_values(result, method, width_metric) / efficiency_scale
            )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
        constrained_layout=True,
    )

    _draw_box_group(
        axes[0],
        coverage_values,
        confounding_strengths,
        ylabel="Coverage",
        title=f"{target_label}: coverage vs confounding strength",
        target_level=1 - alpha,
    )
    axes[0].set_ylim(0.6, 1)

    _draw_box_group(
        axes[1],
        efficiency_values,
        confounding_strengths,
        ylabel="Efficiency (Mbit)",
        title=f"{target_label}: efficiency vs confounding strength",
    )
    axes[1].set_xlabel("Confounding strength")

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"confounding_sweep_{target}_alpha{_fmt_param(alpha)}_eps{_fmt_param(epsilon)}.png",
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path

def main():
    root = _repo_root()
    parser = argparse.ArgumentParser(
        description="Plot coverage and Mbit-normalized efficiency across confounding strengths."
    )
    parser.add_argument("--results-dir", default=os.path.join(root, "results"))
    parser.add_argument("--out-dir", default=os.path.join(root, "plots"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.025)
    parser.add_argument(
        "--target",
        choices=["all", *TARGETS.keys()],
        default="all",
        help="Target to plot. Default creates one plot each for Y_0, Y_1, and ITE.",
    )
    parser.add_argument(
        "--efficiency-scale",
        type=float,
        default=1e6,
        help="Divisor used to convert saved efficiency/width values to Mbit.",
    )
    args = parser.parse_args()

    runs = _load_results(args.results_dir, args.alpha, args.epsilon)
    targets = list(TARGETS.keys()) if args.target == "all" else [args.target]
    for target in targets:
        out_path = plot_confounding_sweep(
            runs,
            target,
            args.out_dir,
            alpha=args.alpha,
            epsilon=args.epsilon,
            efficiency_scale=args.efficiency_scale,
        )
        print(out_path)

if __name__ == "__main__":
    main()
