import argparse
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_experiment import (
    run_experiment,
    fit_base_predictors,
    fit_likelihood_ratio_estimator,
    split_observational_interventional,
    split_train_test_cal,
    build_predictors,
    _json_default,
    _fmt_param,
    _repo_root,
)

EPSILONS = [0.01, 0.025, 0.05, 0.075, 0.1]
CONFOUNDING_STRENGTH = 0.25
ALPHA = 0.1
N_OBS = 2000
N_INT = 50
N_TE = 1000
N_FOLDS = 100
N_TRAIN = 5000
N_TEST_SPLIT = 2000

def main():
    root = _repo_root()
    results_dir = os.path.join(root, "results_eps_sweep")
    os.makedirs(results_dir, exist_ok=True)

    data_path = os.path.join(root, "data", "generated_handover",
                             f"handover_dataset_conf{_fmt_param(CONFOUNDING_STRENGTH)}.npy")
    dataset = np.load(data_path, allow_pickle=True).item()
    dataset = {key: np.asarray(value) for key, value in dataset.items()}

    dataset_obs, dataset_int = split_observational_interventional(dataset)
    dataset_train, dataset_te, dataset_cal = split_train_test_cal(dataset_obs, N_TRAIN, N_TEST_SPLIT)
    predictors = build_predictors(dataset_train, ALPHA)

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
            n_test=N_TE,
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
