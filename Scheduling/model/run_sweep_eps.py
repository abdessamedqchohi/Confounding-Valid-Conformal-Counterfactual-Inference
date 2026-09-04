import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from policy_treatment import load_and_apply_treatment_policy

from run_experiment import (
    run_experiment,
    save_results,
    fit_base_predictors,
    fit_likelihood_ratio_estimator,
    _json_default,
    _slice_dataset,
    _concat_datasets,
    _run_experiment_fold,
    _summarize_fold_logs,
)

EPSILONS = [0.01, 0.025, 0.05, 0.075, 0.1]
CONFOUNDING_STRENGTH = 0.25
ALPHA = 0.1
N_OBS = 2000
N_INT = 50
N_TE = 1000
N_FOLDS = 100

def _fmt_param(value):
    return str(value).replace(".", "p")

def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results_eps_sweep")
    os.makedirs(results_dir, exist_ok=True)

    dataset = load_and_apply_treatment_policy(
        data_path="../data/scheduling_data.npy",
        T=1.,
        confounding_strength=CONFOUNDING_STRENGTH,
        interventional_frac=0.1,
        seed=1,
    )
    mask_int = dataset["is_interventional"] == 1
    mask_obs = dataset["is_interventional"] == 0
    dataset_int = {key: np.asarray(value)[mask_int] for key, value in dataset.items()}
    dataset_obs = {key: np.asarray(value)[mask_obs] for key, value in dataset.items()}

    n_tr = 5000
    n_te_split = 5000
    dataset_train = {key: value[:n_tr] for key, value in dataset_obs.items()}
    dataset_te = {key: value[n_tr:n_tr + n_te_split] for key, value in dataset_obs.items()}
    dataset_cal = {key: value[n_tr + n_te_split:] for key, value in dataset_obs.items()}

    is_RR = dataset_train['T'] == 0
    qr_0_lo, qr_0_hi = fit_base_predictors(dataset_train['X'][is_RR], dataset_train['Y_obs'][is_RR], ALPHA)
    qr_1_lo, qr_1_hi = fit_base_predictors(dataset_train['X'][~is_RR], dataset_train['Y_obs'][~is_RR], ALPHA)
    pr_score_est = fit_likelihood_ratio_estimator(dataset_train['X'], dataset_train['T'])
    predictors = {
        'qr_0_lo': qr_0_lo, 'qr_0_hi': qr_0_hi,
        'qr_1_lo': qr_1_lo, 'qr_1_hi': qr_1_hi,
        'pr_score_est': pr_score_est,
    }

    for eps in EPSILONS:
        print(f"\n{'='*60}")
        print(f"Running eps={eps}")
        print(f"{'='*60}")

        results = run_experiment(
            dataset_cal,
            dataset_int,
            dataset_te,
            predictors,
            dataset_obs_train=dataset_train,
            n_folds=N_FOLDS,
            n_obs=N_OBS,
            n_int=N_INT,
            n_te=N_TE,
            alpha=ALPHA,
            epsilon=eps,
        )

        tag = f"alpha{_fmt_param(ALPHA)}_eps{_fmt_param(eps)}_conf{_fmt_param(CONFOUNDING_STRENGTH)}"
        path = os.path.join(results_dir, f"results_{tag}.json")
        payload = dict(results)
        payload["confounding_strength"] = CONFOUNDING_STRENGTH
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=_json_default)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
