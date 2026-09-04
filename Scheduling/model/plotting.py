import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def _slice_dataset(dataset, indices):
    return {key: np.asarray(value)[indices] for key, value in dataset.items()}

def _concat_datasets(*datasets):
    datasets = [dataset for dataset in datasets if dataset is not None]
    if not datasets:
        raise ValueError("At least one dataset is required.")
    return {
        key: np.concatenate([np.asarray(dataset[key]) for dataset in datasets], axis=0)
        for key in datasets[0].keys()
    }

def _metric_values(results, method, metric):
    json_method = 'GESPI' if method == 'CV-CCI' else method
    return np.asarray([fold['methods'][json_method][metric] for fold in results['folds']], dtype=float)

def _fmt_param(value):
    return str(value).replace(".", "p")

def _with_params(path, alpha, epsilon, confounding_strength):
    if confounding_strength is None:
        return path
    root, ext = os.path.splitext(path)
    tag = f"alpha{_fmt_param(alpha)}_eps{_fmt_param(epsilon)}_conf{_fmt_param(confounding_strength)}"
    return f"{root}_{tag}{ext}"

def plot_results(results, out_path_prefix="../plots/results", alpha=None, epsilon=None, confounding_strength=None):
    all_methods = list(results['folds'][0]['methods'].keys())
    wanted = ['GESPI', 'CCKE', 'Naive', 'WSCP_Inexact', 'WSCP_Exact']
    methods = ['CV-CCI' if m == 'GESPI' else m for m in all_methods if m in wanted]
    targets = [
        ('Y_0', 'cov_0', 'width_0'),
        ('Y_1', 'cov_1', 'width_1'),
        ('ITE', 'cov_ite', 'width_ite'),
    ]
    alpha = results.get('alpha', 0.2) if alpha is None else alpha
    epsilon = results.get('epsilon', 0.05) if epsilon is None else epsilon

    x = np.arange(len(methods))
    box_width = 0.20
    colors = ['#2f6fbb', '#c8524a', '#4f8f5f']
    flierprops = {
        'marker': 'o',
        'markerfacecolor': 'white',
        'markeredgecolor': 'black',
        'markersize': 3.5,
        'alpha': 0.75,
    }
    target_level = 1 - alpha
    gespi_level = 1 - alpha - epsilon

    out_path_combined = _with_params(f"{out_path_prefix}_combined.png", alpha, epsilon, confounding_strength)
    os.makedirs(os.path.dirname(out_path_combined), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    offsets = (np.arange(len(targets)) - 1) * 0.26

    for target_id, (label, cov_key, width_key) in enumerate(targets):
        coverage = [_metric_values(results, method, cov_key) for method in methods]
        widths = [_metric_values(results, method, width_key) for method in methods]
        positions = x + offsets[target_id]

        cov_boxes = axes[0].violinplot(
            coverage,
            positions=positions,
            widths=box_width,
            showmedians=True,
        )
        width_boxes = axes[1].violinplot(
            widths,
            positions=positions,
            widths=box_width,
            showmedians=True,
        )
        for box_group in (cov_boxes, width_boxes):
            for patch in box_group['bodies']:
                patch.set_facecolor(colors[target_id])
                patch.set_alpha(0.55)
                patch.set_edgecolor('black')
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if partname in box_group:
                    box_group[partname].set_edgecolor('black')
                    box_group[partname].set_linewidth(1.2)

        axes[0].plot([], [], color=colors[target_id], linewidth=8, alpha=0.55, label=label)
        axes[1].plot([], [], color=colors[target_id], linewidth=8, alpha=0.55, label=label)

    axes[0].axhline(target_level, color='black', linestyle='--', linewidth=1.2, alpha=0.8, label=f"1-alpha={target_level:.2f}")
    axes[0].axhline(gespi_level, color='#2f6fbb', linestyle='--', linewidth=1.2, alpha=0.8, label=f"GESPI={gespi_level:.2f}")
    axes[0].set_title("Coverage")
    axes[0].set_ylabel("Coverage")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_title("Set Size")
    axes[1].set_ylabel("Average set size")
    for ax in axes:
        ax.set_xlabel("Method")
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.grid(axis='y', alpha=0.25)
        ax.legend(title="Target")
    fig.savefig(out_path_combined, dpi=200)
    plt.close(fig)

    for target_id, (label, cov_key, width_key) in enumerate(targets):
        out_path_single = _with_params(f"{out_path_prefix}_{label}.png", alpha, epsilon, confounding_strength)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        coverage = [_metric_values(results, method, cov_key) for method in methods]
        widths = [_metric_values(results, method, width_key) for method in methods]

        cov_boxes = axes[0].violinplot(
            coverage,
            positions=x,
            widths=0.4,
            showmedians=True,
        )
        width_boxes = axes[1].violinplot(
            widths,
            positions=x,
            widths=0.4,
            showmedians=True,
        )

        for box_group in (cov_boxes, width_boxes):
            for patch in box_group['bodies']:
                patch.set_facecolor(colors[target_id])
                patch.set_alpha(0.55)
                patch.set_edgecolor('black')
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
                if partname in box_group:
                    box_group[partname].set_edgecolor('black')
                    box_group[partname].set_linewidth(1.2)

        axes[0].axhline(target_level, color='black', linestyle='--', linewidth=1.2, alpha=0.8, label=f"1-alpha={target_level:.2f}")
        axes[0].axhline(gespi_level, color='#2f6fbb', linestyle='--', linewidth=1.2, alpha=0.8, label=f"GESPI={gespi_level:.2f}")
        axes[0].set_title(f"Coverage ({label})")
        axes[0].set_ylabel("Coverage")
        axes[0].set_ylim(0, 1.05)
        axes[1].set_title(f"Set Size ({label})")
        axes[1].set_ylabel("Average set size")

        for ax in axes:
            ax.set_xlabel("Method")
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=30, ha="right")
            ax.grid(axis='y', alpha=0.25)
            ax.legend()

        fig.savefig(out_path_single, dpi=200)
        plt.close(fig)

    return out_path_combined

def _fit_target_prediction_sets(
    dataset_obs_cal,
    dataset_int_cal,
    dataset_te,
    predictors,
    method_classes,
    dataset_obs_train=None,
    target=1,
    alpha=0.2,
    epsilon=0.05,
):
    id_tgt_obs = dataset_obs_cal['T'] == target
    id_tgt_int = dataset_int_cal['T'] == target

    X_obs = dataset_obs_cal['X'][id_tgt_obs].astype(float)
    Y_obs = dataset_obs_cal['Y_obs'][id_tgt_obs].astype(float)
    X_int = dataset_int_cal['X'][id_tgt_int].astype(float)
    Y_int = dataset_int_cal['Y_obs'][id_tgt_int].astype(float)
    X_te = dataset_te['X'].astype(float)

    dataset_obs_wscp = (
        _concat_datasets(dataset_obs_train, dataset_obs_cal)
        if dataset_obs_train is not None
        else dataset_obs_cal
    )

    X_obs_all = dataset_obs_wscp['X'].astype('float')
    T_obs_all = dataset_obs_wscp['T'].astype(int)
    Y_obs_all = dataset_obs_wscp['Y_obs'].astype('float')

    X_int_all = dataset_int_cal['X'].astype('float')
    T_int_all = dataset_int_cal['T'].astype(int)
    Y_int_all = dataset_int_cal['Y_obs'].astype('float')

    gespi = method_classes['GESPI'](
        alpha=alpha,
        epsilon=epsilon,
        q_lo=predictors[f'qr_{target}_lo'],
        q_hi=predictors[f'qr_{target}_hi'],
        w=predictors['pr_score_est'],
    )
    gespi.fit(X_obs, Y_obs, X_int, Y_int, target)

    ccke = method_classes['CCKE'](
        alpha=alpha,
        q_lo=predictors[f'qr_{target}_lo'],
        q_hi=predictors[f'qr_{target}_hi'],
        w=predictors['pr_score_est'],
    )
    ccke.fit(X_obs, Y_obs, target)

    nccke = method_classes['NCCKE'](
        alpha=alpha,
        q_lo=predictors[f'qr_{target}_lo'],
        q_hi=predictors[f'qr_{target}_hi'],
    )
    nccke.fit(X_obs, Y_obs, target)

    naive = method_classes['Naive'](
        alpha=alpha,
        q_lo=predictors[f'qr_{target}_lo'],
        q_hi=predictors[f'qr_{target}_hi'],
    )
    naive.fit(X_int, Y_int, target)

    cke = method_classes['CKE'](
        q_lo=predictors[f'qr_{target}_lo'],
        q_hi=predictors[f'qr_{target}_hi'],
    )

    wscp_in = method_classes['WSCP_Inexact'](
        alpha=alpha,
        q_lo=predictors[f'qr_{target}_lo'],
        q_hi=predictors[f'qr_{target}_hi'],
        w_pred=predictors['pr_score_est'],
    )
    wscp_in.fit(X_obs_all, T_obs_all, Y_obs_all, X_int_all, T_int_all, Y_int_all, target)

    wscp_ex = method_classes['WSCP_Exact'](
        alpha=alpha,
        q_lo=predictors[f'qr_{target}_lo'],
        q_hi=predictors[f'qr_{target}_hi'],
        w_pred=predictors['pr_score_est'],
    )
    wscp_ex.fit(X_obs_all, T_obs_all, Y_obs_all, X_int_all, T_int_all, Y_int_all, target)

    return {
        'GESPI': gespi.predict_set(X_te),
        'CCKE': ccke.predict_set(X_te),
        'NCCKE': nccke.predict_set(X_te),
        'Naive': naive.predict_set(X_te),
        'CKE': cke.predict_set(X_te),
        'WSCP_Inexact': wscp_in.predict_set(X_te),
        'WSCP_Exact': wscp_ex.predict_set(X_te),
    }

def plot_treatment_realization_intervals(
    dataset_obs_cal,
    dataset_int,
    dataset_plot,
    predictors,
    method_classes,
    dataset_obs_train=None,
    n=10,
    target=1,
    seed=0,
    out_dir=None,
    alpha=0.2,
    epsilon=0.05,
    confounding_strength=None,
):
    rng = np.random.default_rng(seed)

    if target in [0, 1]:
        target_mask = dataset_plot['T'] == target
        candidate_idx = np.flatnonzero(target_mask)
        if len(candidate_idx) == 0:
            raise ValueError(f"No observational plotting samples found with T == {target}.")
    else:
        candidate_idx = np.arange(len(dataset_plot['T']))

    chosen_idx = rng.choice(candidate_idx, size=min(n, len(candidate_idx)), replace=False)
    dataset_te = _slice_dataset(dataset_plot, chosen_idx)

    def get_pred_sets(tgt):
        return _fit_target_prediction_sets(
            dataset_obs_cal, dataset_int, dataset_te, predictors, method_classes,
            dataset_obs_train=dataset_obs_train, target=tgt, alpha=alpha, epsilon=epsilon
        )

    sets_0 = get_pred_sets(0) if target in [0, 'ITE', 'all'] else None
    sets_1 = get_pred_sets(1) if target in [1, 'ITE', 'all'] else None

    def get_ite_sets(s0, s1):
        def _hull(pred_set):
            if len(pred_set) == 2:
                return pred_set[0], pred_set[1]
            L = np.minimum(pred_set[0], pred_set[2])
            U = np.maximum(pred_set[1], pred_set[3])
            return L, U

        ite_sets = {}
        for m in s0.keys():
            L_0, U_0 = _hull(s0[m])
            L_1, U_1 = _hull(s1[m])
            ite_sets[m] = (L_1 - U_0, U_1 - L_0)
        return ite_sets

    sets_ite = get_ite_sets(sets_0, sets_1) if target in ['ITE', 'all'] else None

    methods = ['CV-CCI', 'CCKE', 'Naive', 'WSCP_Inexact', 'WSCP_Exact']
    colors = {
        'CV-CCI': '#2f6fbb',
        'CCKE': '#c8524a',
        'NCCKE': '#4f8f5f',
        'Naive': '#5c8a8a',
        'CKE': '#7b4fa1',
        'WSCP_Inexact': '#e68a1a',
        'WSCP_Exact': '#8b6914',
    }
    offsets = np.linspace(-0.24, 0.24, len(methods))
    users = np.arange(dataset_te['Y0_RR'].shape[1])

    if out_dir is None:
        if confounding_strength is None:
            out_dir = f"../plots/treatment_{target}_intervals"
        else:
            out_dir = (
                f"../plots/treatment_{target}_intervals_"
                f"alpha{_fmt_param(alpha)}_eps{_fmt_param(epsilon)}_conf{_fmt_param(confounding_strength)}"
            )
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []

    for local_id, original_idx in enumerate(chosen_idx):
        if target == 'all':
            fig, axs = plt.subplots(1, 3, figsize=(24, 6), constrained_layout=True)
        else:
            fig, ax_single = plt.subplots(figsize=(10, 5), constrained_layout=True)
            axs = [ax_single]

        plot_configs = []
        if target == 0 or target == 'all':
            plot_configs.append({
                'ax': axs[0] if target == 'all' else axs[0],
                'true': dataset_te['Y0_RR'][local_id].astype(float),
                'sets': sets_0,
                'title': f"T=0 (RR) prediction intervals, sample {int(original_idx)}"
            })
        if target == 1 or target == 'all':
            plot_configs.append({
                'ax': axs[1] if target == 'all' else axs[0],
                'true': dataset_te['Y1_PFCA'][local_id].astype(float),
                'sets': sets_1,
                'title': f"T=1 (PFCA) prediction intervals, sample {int(original_idx)}"
            })
        if target == 'ITE' or target == 'all':
            idx = 2 if target == 'all' else 0
            plot_configs.append({
                'ax': axs[idx],
                'true': dataset_te['Y1_PFCA'][local_id].astype(float) - dataset_te['Y0_RR'][local_id].astype(float),
                'sets': sets_ite,
                'title': f"ITE prediction intervals, sample {int(original_idx)}"
            })

        for config in plot_configs:
            ax = config['ax']
            true_vals = config['true']
            pred_sets = config['sets']

            tick_half_height = 0.18
            ax.vlines(
                true_vals,
                users - tick_half_height,
                users + tick_half_height,
                color='black',
                linewidth=3,
                zorder=6,
                label='True Outcome',
            )

            for method_id, method in enumerate(methods):
                pred_set = pred_sets[method]
                method_y = users + offsets[method_id]
                if len(pred_set) == 2:
                    intervals = [(pred_set[0][local_id], pred_set[1][local_id])]
                else:
                    intervals = [
                        (pred_set[0][local_id], pred_set[1][local_id]),
                        (pred_set[2][local_id], pred_set[3][local_id]),
                    ]

                first_segment = True
                for lower, upper in intervals:
                    valid = np.isfinite(lower) & np.isfinite(upper) & (lower <= upper)
                    ax.hlines(
                        method_y[valid],
                        lower[valid],
                        upper[valid],
                        color=colors[method],
                        linewidth=5,
                        alpha=0.55,
                        label=method if first_segment else None,
                    )
                    ax.scatter(
                        lower[valid],
                        method_y[valid],
                        color=colors[method],
                        marker='|',
                        s=70,
                        alpha=0.9,
                    )
                    ax.scatter(
                        upper[valid],
                        method_y[valid],
                        color=colors[method],
                        marker='|',
                        s=70,
                        alpha=0.9,
                    )
                    first_segment = False

            ax.set_title(config['title'])
            ax.set_xlabel("Queue length / ITE")
            ax.set_ylabel("User")
            ax.set_yticks(users)
            ax.set_yticklabels([str(user) for user in users])
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.25)

            if config == plot_configs[0]:
                ax.legend(ncol=7, loc='upper center', bbox_to_anchor=(0.5, -0.12))

        if confounding_strength is None:
            filename = f"target_{target}_sample_{local_id:02d}.png"
        else:
            filename = (
                f"target_{target}_sample_{local_id:02d}_"
                f"alpha{_fmt_param(alpha)}_eps{_fmt_param(epsilon)}_conf{_fmt_param(confounding_strength)}.png"
            )
        out_path = os.path.join(out_dir, filename)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        out_paths.append(out_path)

    return out_paths
