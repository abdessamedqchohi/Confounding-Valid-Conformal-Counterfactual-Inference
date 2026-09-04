import argparse
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
    "WSCP_Exact":   "#c8a243",
    "WSCP_Inexact": "#e68a00",
    "Naive":        "#5c8a8a",
}

MARKERS = {
    "GESPI":        "o",
    "CCKE":         "s",
    "WSCP_Exact":   "D",
    "WSCP_Inexact": "*",
    "Naive":        "^",
}

TARGETS = {
    "Y_0":  ("cov_0",   "width_0",   r"$Y(0)$"),
    "Y_1":  ("cov_1",   "width_1",   r"$Y(1)$"),
    "ITE":  ("cov_ite", "width_ite", r"ITE"),
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
        [fold["methods"][method][metric] for fold in result["folds"]],
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

def _draw_box_group(ax, grouped_values, confounding_strengths,
                   ylabel, target_level=None, guardrail_level=None):
    centers  = np.arange(len(confounding_strengths), dtype=float)
    n_methods = len(METHODS)
    offsets   = np.linspace(-0.24, 0.24, n_methods)
    box_width = 0.12
    flierprops = {
        "marker":          "o",
        "markerfacecolor": "white",
        "markeredgecolor": "black",
        "markersize":      3.5,
        "alpha":           0.75,
    }

    handles = []
    for method_id, method in enumerate(METHODS):
        values = grouped_values[method]          
        clean_values = [v[~np.isnan(v)] if len(v)>0 else v for v in values]
        parts = ax.violinplot(
            clean_values,
            positions=centers + offsets[method_id],
            widths=box_width,
            showmeans=False,
            showmedians=True,
            showextrema=True
        )
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS[method])
            pc.set_edgecolor("black")
            pc.set_alpha(0.58)
            pc.set_linewidth(1.0)
            
        for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
            vp = parts[partname]
            vp.set_edgecolor('black')
            vp.set_linewidth(1.2)
            
        medians_y = [np.median(v) if len(v)>0 else np.nan for v in clean_values]
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

    if guardrail_level is not None:
        h_guardrail = ax.axhline(
            guardrail_level,
            color="#2f6fbb", linestyle=":", linewidth=1.1, alpha=0.9,
            label=rf"$1-\alpha-\epsilon={guardrail_level:.3f}$",
        )
        handles.append(h_guardrail)

    ax.set_ylabel(ylabel)
    ax.set_xticks(centers)
    mapping = {0.0: "0", 0.25: "2", 0.5: "4", 0.75: "6", 1.0: "8"}
    ax.set_xticklabels([rf"${mapping.get(float(c), c)}$" for c in confounding_strengths])
    ax.set_xlabel(r"$\lambda$", labelpad=3)
    ax.grid(False)
    return handles

def plot_sweep(runs, target, out_dir, alpha=0.1, epsilon=0.025):
    cov_metric, width_metric, target_label = TARGETS[target]
    confs, cov_data, width_data = _collect_fold_arrays(
        runs, METHODS, cov_metric, width_metric)

    plt.rcParams.update({
        "text.usetex":       True,
        "font.family":       "serif",
        "font.size":         18,
        "axes.titlesize":    20,
        "axes.labelsize":    20,
        "legend.fontsize":   18,
        "xtick.labelsize":   18,
        "ytick.labelsize":   18,
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

    fig, (ax_cov, ax_eff) = plt.subplots(
        1, 2,
        figsize=(14, 4.5),
        constrained_layout=True,
    )

    handles = _draw_box_group(
        ax_cov, cov_data, confs,
        ylabel="Coverage",
        target_level=1 - alpha,
        guardrail_level=1 - alpha - epsilon,
    )
    ax_cov.set_ylim(0.60, 1.02)

    _draw_box_group(
        ax_eff, width_data, confs,
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
        f"sweep_{target}_alpha{_fmt_param(alpha)}_eps{_fmt_param(epsilon)}_violin.pdf",
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight", bbox_extra_artists=(leg,))
    fig.savefig(out_path.replace(".pdf", ".png"), dpi=200, bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)
    return out_path

def main():
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Paper-style confounding sweep plot.")
    parser.add_argument("--results-dir", default=os.path.join(root, "results"))
    parser.add_argument("--out-dir",     default=os.path.join(root, "plots"))
    parser.add_argument("--alpha",       type=float, default=0.1)
    parser.add_argument("--epsilon",     type=float, default=0.025)
    parser.add_argument(
        "--target",
        choices=["all", *TARGETS.keys()],
        default="all",
    )
    args = parser.parse_args()

    runs = _load_results(args.results_dir, args.alpha, args.epsilon)
    targets = list(TARGETS.keys()) if args.target == "all" else [args.target]
    for t in targets:
        path = plot_sweep(runs, t, args.out_dir, alpha=args.alpha, epsilon=args.epsilon)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
