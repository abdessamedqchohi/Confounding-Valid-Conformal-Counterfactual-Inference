import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

METHODS = ["GESPI", "Naive"]

DISPLAY_NAMES = {
    "GESPI": "CV-CCI",
    "Naive":  "Guardrail",
}

COLORS = {
    "GESPI": "#2f6fbb",
    "Naive":  "#5c8a8a",
}

MARKERS = {
    "GESPI": "o",
    "Naive":  "^",
}

TARGETS = {
    "RR":  ("cov_0",   "width_0",   r"RR ($Y_0$)"),
    "PFCA": ("cov_1",   "width_1",   r"PFCA ($Y_1$)"),
    "ITE":  ("cov_ite", "width_ite", r"ITE"),
}

ALPHA = 0.1
CONFOUNDING_STRENGTH = 0.25
EPSILONS = [0.01, 0.025, 0.05, 0.075, 0.1]

def _fmt_param(value):
    return str(value).replace(".", "p")

def _fold_values(result, method, metric):
    vals = np.asarray(
        [fold["methods"][method][metric] for fold in result["folds"]],
        dtype=float,
    )
    vals = np.where(np.isinf(vals), np.nan, vals)
    return vals

def _load_eps_results(results_dir):
    runs = []
    for eps in EPSILONS:
        tag = f"alpha{_fmt_param(ALPHA)}_eps{_fmt_param(eps)}_conf{_fmt_param(CONFOUNDING_STRENGTH)}"
        path = os.path.join(results_dir, f"results_{tag}.json")
        if not os.path.exists(path):
            print(f"Warning: missing {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            result = json.load(f)
        runs.append((eps, result))
    return runs

def _collect_fold_arrays(runs, methods, cov_metric, width_metric):
    epsilons = [e for e, _ in runs]
    cov_data = {m: [] for m in methods}
    width_data = {m: [] for m in methods}
    for _, result in runs:
        for m in methods:
            cv = _fold_values(result, m, cov_metric)
            wv = _fold_values(result, m, width_metric)
            cov_data[m].append(cv)
            width_data[m].append(wv)
    return epsilons, cov_data, width_data

def _draw_box_group(ax, grouped_values, x_values, methods, ylabel,
                    target_level=None, show_guardrail_lines=False):
    centers = np.arange(len(x_values), dtype=float)
    n_methods = len(methods)
    offsets = np.linspace(-0.18, 0.18, n_methods)
    box_width = 0.20
    flierprops = {
        "marker": "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize": 3.5,
        "alpha": 0.75,
    }

    handles = []
    for method_id, method in enumerate(methods):
        values = grouped_values[method]
        clean_values = [v[~np.isnan(v)] if len(v) > 0 else v for v in values]
        parts = ax.boxplot(
            clean_values,
            positions=centers + offsets[method_id],
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            flierprops=flierprops,
        )
        for pc in parts['boxes']:
            pc.set_facecolor(COLORS[method])
            pc.set_edgecolor("black")
            pc.set_alpha(0.58)
            pc.set_linewidth(1.0)
        for median in parts['medians']:
            median.set_color('black')
            median.set_linewidth(1.2)
        for partname in ('whiskers', 'caps'):
            for item in parts[partname]:
                item.set_color('black')
                item.set_linewidth(1.2)

        medians_y = [np.nanmedian(v) if len(v) > 0 else np.nan for v in clean_values]
        ax.plot(centers + offsets[method_id], medians_y,
                marker=MARKERS[method], linestyle="None",
                markerfacecolor=COLORS[method], markeredgecolor="black",
                markersize=9, zorder=3)

        h = Line2D([0], [0],
            marker=MARKERS[method], color="w",
            markerfacecolor=COLORS[method], markeredgecolor="black",
            markersize=9, label=DISPLAY_NAMES[method],
        )
        handles.append(h)

    if target_level is not None:
        h_target = ax.axhline(
            target_level,
            color="black", linestyle="--", linewidth=1.1, alpha=0.75,
            label=rf"$1-\alpha={target_level:.2f}$",
        )
        handles.append(h_target)

    if show_guardrail_lines:
        for eps_val in x_values:
            guard_level = 1 - ALPHA - eps_val
            idx = x_values.index(eps_val)
            ax.plot([idx - 0.3, idx + 0.3], [guard_level, guard_level],
                    color="#2f6fbb", linestyle=":", linewidth=1.0, alpha=0.6)

    ax.set_ylabel(ylabel)
    ax.set_xticks(centers)
    ax.set_xticklabels([rf"${e}$" for e in x_values])
    ax.set_xlabel(r"$\epsilon$", labelpad=3)
    ax.grid(False)
    return handles

def plot_eps_sweep(runs, target, out_dir):
    cov_metric, width_metric, target_label = TARGETS[target]
    epsilons, cov_data, width_data = _collect_fold_arrays(
        runs, METHODS, cov_metric, width_metric)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 18,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "legend.fontsize": 18,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, (ax_cov, ax_eff) = plt.subplots(
        1, 2,
        figsize=(14, 4.5),
        constrained_layout=True,
    )

    handles = _draw_box_group(
        ax_cov, cov_data, epsilons, METHODS,
        ylabel="Coverage",
        target_level=1 - ALPHA,
        show_guardrail_lines=True,
    )
    ax_cov.set_ylim(0.60, 1.02)

    _draw_box_group(
        ax_eff, width_data, epsilons, METHODS,
        ylabel="Efficiency",
    )

    leg = fig.legend(
        handles=handles,
        loc="center left",
        ncol=1,
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
        fontsize=18,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir,
        f"sweep_eps_{target}.pdf",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", bbox_extra_artists=(leg,))
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)
    return out_path

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(script_dir)
    results_dir = os.path.join(root, "results_eps_sweep")
    out_dir = os.path.join(root, "plots_eps_sweep")

    runs = _load_eps_results(results_dir)
    if not runs:
        print("No results found. Run run_sweep_eps.py first.")
        return

    for target in TARGETS:
        path = plot_eps_sweep(runs, target, out_dir)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
