import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "Times", "Palatino", "serif"],
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18,
})

def _metric_values(results, method, metric):
    return np.asarray([fold['methods'][method][metric] for fold in results['folds'] if method in fold['methods']], dtype=float)

def load_data():
    confs = [0, 0.25, 0.5, 0.75, 1]
    alpha = 0.1
    epsilon = 0.025
    
    targets = {
        'RR': ('cov_0', 'width_0'),
        'PFCA': ('cov_1', 'width_1'),
        'ITE': ('cov_ite', 'width_ite')
    }
    methods = {
        'CV-CCI': 'GESPI',
        'CCKE': 'CCKE',
        'wSCP-DR exact': 'WSCP_Exact',
        'wSCP-DR inexact': 'WSCP_Inexact',
        'Guardrail': 'Naive'
    }
    
    data = {'coverage': {t: {c: {m: [] for m in methods} for c in confs} for t in targets},
            'width': {t: {c: {m: [] for m in methods} for c in confs} for t in targets}}
    
    for c in confs:
        c_str = str(c).replace('.', 'p')
        if c == 1: c_str = '1'
        if c == 0: c_str = '0'
        
        file_path = f"results/results_alpha0p1_eps0p025_conf{c_str}.json"
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            res = json.load(f)
            
        for t_name, (cov_key, width_key) in targets.items():
            for m_label, m_key in methods.items():
                cov_vals = _metric_values(res, m_key, cov_key)
                width_vals = _metric_values(res, m_key, width_key)
                
                data['coverage'][t_name][c][m_label] = cov_vals
                data['width'][t_name][c][m_label] = width_vals
                
    return data, confs, methods, targets, alpha, epsilon

def plot_metric(ax, metric_type, t_name, data, confs, methods, colors, markers, alpha, epsilon):
    n_methods = len(methods)
    width = 0.15
    target_level = 1 - alpha
    gespi_level = 1 - alpha - epsilon

    for i, c in enumerate(confs):
        for j, (m_label, m_key) in enumerate(methods.items()):
            vals = data[metric_type][t_name][c][m_label]
            if len(vals) == 0: continue
            
            if metric_type == 'width':
                vals = vals[np.isfinite(vals)]
                vals = [v / 1000.0 for v in vals]
                if len(vals) == 0:
                    continue
            
            pos = i + (j - n_methods/2 + 0.5) * width
            
            bp = ax.boxplot(vals, positions=[pos], widths=width*0.8, patch_artist=True,
                            showfliers=True, 
                            flierprops=dict(marker='o', markerfacecolor='none', 
                                            markeredgecolor='black', markersize=4, alpha=0.5))
            
            for patch in bp['boxes']:
                patch.set_facecolor(colors[m_label])
                patch.set_alpha(0.8)
                patch.set_edgecolor('black')
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)
                
            median_val = np.median(vals)
            ax.scatter(pos, median_val, color=colors[m_label], marker=markers[m_label], 
                       s=150, edgecolors='black', zorder=4, linewidths=1.5)
            
            if i == 0:
                ax.plot([], [], color=colors[m_label], marker=markers[m_label], 
                        label=m_label, linestyle='None', markersize=8, markeredgecolor='black')
                
    ax.set_xticks(range(len(confs)))
    ax.set_xticklabels(confs)
    ax.set_xlabel(r"$\gamma$")
    
    if metric_type == 'coverage':
        ax.set_ylabel("Coverage")
        ax.axhline(target_level, color='black', linestyle='--', label=f'1 - $\\alpha$ = {target_level:.3f}')
        ax.axhline(gespi_level, color='#2f6fbb', linestyle=':', label=f'1 - $\\alpha$ - $\\epsilon$ = {gespi_level:.3f}')
        ax.set_ylim(0.5, 1.05)
    else:
        ax.set_ylabel("Efficiency")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_sweep():
    data, confs, methods, targets, alpha, epsilon = load_data()
    os.makedirs('plots_sweep', exist_ok=True)
    
    colors = {
        'CV-CCI': '#2f6fbb',
        'CCKE': '#c8524a',
        'wSCP-DR exact': '#8b6914',
        'wSCP-DR inexact': '#e68a1a',
        'Guardrail': '#5c8a8a'
    }
    markers = {
        'CV-CCI': 'o',
        'CCKE': 's',
        'wSCP-DR exact': 'D',
        'wSCP-DR inexact': '*',
        'Guardrail': '^'
    }
    
    title_map = {
        'RR': 'RR (Round Robin)',
        'PFCA': 'PFCA (Proportional Fair Channel Aware)',
        'ITE': 'ITE (Individual Treatment Effect)'
    }

    for t_name in targets:
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        for ax_idx, metric_type in enumerate(['coverage', 'width']):
            plot_metric(axes[ax_idx], metric_type, t_name, data, confs, methods, colors, markers, alpha, epsilon)
            
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
        plt.tight_layout()
        plt.savefig(f"plots_sweep/combined_{t_name}.pdf", bbox_inches='tight')
        plt.savefig(f"plots_sweep/combined_{t_name}.png", dpi=200, bbox_inches='tight')
        plt.close()

        fig_cov, ax_cov = plt.subplots(figsize=(8, 6))
        plot_metric(ax_cov, 'coverage', t_name, data, confs, methods, colors, markers, alpha, epsilon)
        handles, labels = ax_cov.get_legend_handles_labels()
        ax_cov.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))
        fig_cov.suptitle(title_map.get(t_name, t_name), fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"plots_sweep/coverage_{t_name}.pdf", bbox_inches='tight')
        plt.savefig(f"plots_sweep/coverage_{t_name}.png", dpi=200, bbox_inches='tight')
        plt.close()

        fig_width, ax_width = plt.subplots(figsize=(8, 6))
        plot_metric(ax_width, 'width', t_name, data, confs, methods, colors, markers, alpha, epsilon)
        handles, labels = ax_width.get_legend_handles_labels()
        ax_width.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0.5))
        fig_width.suptitle(title_map.get(t_name, t_name), fontsize=16, y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"plots_sweep/width_{t_name}.pdf", bbox_inches='tight')
        plt.savefig(f"plots_sweep/width_{t_name}.png", dpi=200, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    plot_sweep()
