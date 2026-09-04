import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

METHODS = ["GESPI", "CCKE", "WSCP_Exact", "WSCP_Inexact", "Naive"]

DISPLAY_NAMES = {
    "GESPI":        "CV-CCI",
    "CCKE":         "CCKE",
    "WSCP_Exact":   "wSCP-DR exact",
    "WSCP_Inexact": "wSCP-DR inexact",
    "Naive":        "Guardrail",
}

COLORS = {
    "GESPI":        "#2f6fbb",
    "CCKE":         "#c8524a",
    "WSCP_Exact":   "#b8860b",
    "WSCP_Inexact": "#e68a00",
    "Naive":        "#5c8a8a",
}

MARKERS = {
    "GESPI":        "o",
    "CCKE":         "s",
    "WSCP_Exact":   "D",
    "WSCP_Inexact": "P",
    "Naive":        "^",
}

TARGETS = {
    "Y_0":  ("cov_0",   "width_0",   r"$Y(0)$"),
    "Y_1":  ("cov_1",   "width_1",   r"$Y(1)$"),
    "ITE":  ("cov_ite", "width_ite", r"ITE"),
}

def setup_latex_style():
    plt.rcParams.update({
        "text.usetex":       True,
        "font.family":       "serif",
        "font.serif":        ["Computer Modern Roman", "Times New Roman", "DejaVu Serif", "serif"],
        "mathtext.fontset":  "cm",
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    12,
        "legend.fontsize":   10.0,
        "xtick.labelsize":   10.5,
        "ytick.labelsize":   10.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         False,
    })

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
        conf = result.get("confounding_strength")
        if conf is None:
            conf = _parse_conf_from_path(path)
        if conf is None:
            conf = 0.0
        runs.append((float(conf), result))
    runs.sort(key=lambda x: x[0])
    if not runs:
        raise ValueError(f"No result files found in {results_dir!r}.")
    return runs

def _fold_values(result, method, metric):
    vals = np.asarray(
        [fold["methods"][method][metric] for fold in result["folds"] if method in fold["methods"]],
        dtype=float,
    )
    vals = np.where(np.isinf(vals), np.nan, vals)
    return vals

def _collect_fold_arrays(runs, methods, cov_metric, width_metric):
    confs = [c for c, _ in runs]
    cov_data   = {m: [] for m in methods}
    width_data = {m: [] for m in methods}
    for _, result in runs:
        for m in methods:
            cv = _fold_values(result, m, cov_metric)
            wv = _fold_values(result, m, width_metric)
            cov_data[m].append(cv)
            width_data[m].append(wv)
    return confs, cov_data, width_data

def _draw_box_group(ax, grouped_values, confounding_strengths, ylabel):
    centers   = np.arange(len(confounding_strengths), dtype=float)
    n_methods = len(METHODS)
    offsets   = np.linspace(-0.24, 0.24, n_methods)
    box_width = 0.10

    flierprops = {
        "marker":          "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize":      3.0,
        "alpha":           0.6,
    }

    for method_id, method in enumerate(METHODS):
        values = grouped_values[method]
        clean_values = [v[~np.isnan(v)] if len(v) > 0 else v for v in values]
        parts = ax.boxplot(
            clean_values,
            positions=centers + offsets[method_id],
            widths=box_width,
            patch_artist=True,
            showfliers=True,
            flierprops=flierprops
        )
        for pc in parts['boxes']:
            pc.set_facecolor(COLORS[method])
            pc.set_edgecolor("black")
            pc.set_alpha(0.55)
            pc.set_linewidth(0.8)
            
        for median in parts['medians']:
            median.set_color('black')
            median.set_linewidth(1.1)
        for partname in ('whiskers', 'caps'):
            for item in parts[partname]:
                item.set_color('black')
                item.set_linewidth(0.8)
            
        medians_y = [np.median(v) if len(v) > 0 else np.nan for v in clean_values]
        ax.plot(
            centers + offsets[method_id], medians_y,
            marker=MARKERS[method], linestyle="None",
            markerfacecolor=COLORS[method], markeredgecolor="black",
            markersize=6.0, markeredgewidth=0.8, zorder=4
        )

    ax.set_ylabel(ylabel)
    ax.set_xticks(centers)
    mapping = {0.0: "0", 0.25: "2", 0.5: "4", 0.75: "6", 1.0: "8"}
    ax.set_xticklabels([rf"${mapping.get(float(c), c)}$" for c in confounding_strengths])
    ax.set_xlabel(r"$\lambda$", labelpad=0)
    ax.grid(axis="y", linestyle=":", alpha=0.4, color="gray")

def plot_sweep(runs, target, out_dirs, alpha=0.1, epsilon=0.025):
    setup_latex_style()
    cov_metric, width_metric, target_label = TARGETS[target]
    confs, cov_data, width_data = _collect_fold_arrays(
        runs, METHODS, cov_metric, width_metric)

    target_level = 1 - alpha
    guardrail_level = 1 - alpha - epsilon

    fig, (ax_cov, ax_eff) = plt.subplots(
        1, 2,
        figsize=(7.0, 2.6),
    )

    _draw_box_group(
        ax_cov, cov_data, confs,
        ylabel="Coverage",
    )
    h_target = ax_cov.axhline(
        target_level,
        color="black", linestyle="--", linewidth=1.1, alpha=0.85,
        label=rf"$1-\alpha = {target_level:.2f}$",
    )
    h_guardrail = ax_cov.axhline(
        guardrail_level,
        color="#2f6fbb", linestyle=":", linewidth=1.2, alpha=0.9,
        label=rf"$1-\alpha-\epsilon = {guardrail_level:.3f}$",
    )
    ax_cov.set_ylim(0.55, 1.03)

    _draw_box_group(
        ax_eff, width_data, confs,
        ylabel="Efficiency",
    )

    method_handles = [
        Line2D([0], [0],
               marker=MARKERS[m], color="w",
               markerfacecolor=COLORS[m], markeredgecolor="black",
               markersize=6.5, label=DISPLAY_NAMES[m])
        for m in METHODS
    ]
    all_handles = method_handles + [h_target, h_guardrail]

    fig.subplots_adjust(bottom=0.25, wspace=0.26, left=0.09, right=0.98, top=0.94)
    leg = fig.legend(
        handles=all_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.125),
        ncol=7,
        frameon=True,
        edgecolor="#333333",
        facecolor="#ffffff",
        framealpha=1.0,
        fontsize=10.0,
        handletextpad=0.3,
        columnspacing=0.55,
        borderpad=0.25,
    )

    filename_base = f"sweep_{target}_alpha{_fmt_param(alpha)}_eps{_fmt_param(epsilon)}_boxplot"
    
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)
        pdf_path = os.path.join(d, f"{filename_base}.pdf")
        png_path = os.path.join(d, f"{filename_base}.png")
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg,))
        fig.savefig(png_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg,))
        print(f"Saved: {pdf_path}")

    plt.close(fig)

def main():
    root = _repo_root()
    results_dir = os.path.join(root, "results")
    alpha = 0.1
    epsilon = 0.025

    out_dirs = [
        os.path.join(root, "plots_boxplot"),
        os.path.join(root, "plots_double_col"),
    ]

    runs = _load_results(results_dir, alpha, epsilon)
    for target in TARGETS:
        plot_sweep(runs, target, out_dirs, alpha=alpha, epsilon=epsilon)

if __name__ == "__main__":
    main()
