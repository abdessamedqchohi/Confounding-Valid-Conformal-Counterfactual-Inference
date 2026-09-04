import argparse
import os

import numpy as np

bw_mhz = 5
Tmax = 80
num_user = 8

def get_cqi_payload(min_cqi):
    se_map = [
        0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141,
        2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547, 9.6,
    ]
    return se_map[int(min_cqi)] * bw_mhz / num_user * 1E3

def calculate_p_rr(s_backlog, cqi_values, delays, n_prbs, T, strength=1):
    strength = float(np.clip(strength, 0.0, 1.0))
    tx_data_bits = np.array([get_cqi_payload(c) for c in cqi_values])
    input_val = np.max(((s_backlog - (tx_data_bits * Tmax / num_user)) / tx_data_bits))
    frac_ue_prbs = num_user - n_prbs
    p_t_u = 1. / (1 + np.exp(frac_ue_prbs/T))
    p_t_x = 1. / (1 + np.exp(-input_val/T))
    eps = 0.1
    return np.clip(p_t_u*strength+p_t_x*(1-strength), eps, 1 - eps)

def apply_treatment_policy(raw_dataset, T=10, confounding_strength=1, interventional_frac=0.05, seed=None):
    X_raw = raw_dataset["X"]
    Y0_RR = np.asarray(raw_dataset["Y0_RR"])
    Y1_PFCA = np.asarray(raw_dataset["Y1_PFCA"])
    n_episodes = len(X_raw)
    rng = np.random.default_rng(seed)
    treatments = []
    modes = []
    p_rrs = []
    for i in range(n_episodes):
        X_backlog = np.sum(X_raw[i]['s'], axis=1)
        X_cqi = X_raw[i]['cqi']
        X_delays = np.sum(X_raw[i]['e'], axis=1)
        U_n_prbs = X_raw[i]['prbs']
        p_rr = calculate_p_rr(
            X_backlog,
            X_cqi,
            X_delays,
            U_n_prbs,
            T=T,
            strength=confounding_strength,
        )

        if i < int(interventional_frac * n_episodes):
            mode = 1
            choice = 0 if rng.random() < 0.5 else 1
        else:
            mode = 0
            choice = 0 if rng.random() < p_rr else 1

        treatments.append(choice)
        modes.append(mode)
        p_rrs.append(p_rr)

    treatments = np.asarray(treatments)
    Y_obs = np.where((treatments == 0)[:, None], Y0_RR, Y1_PFCA)
    X = np.asarray([np.concatenate((np.sum(x['s'], axis=1),x['cqi'],np.sum(x['e'], axis=1))) for x in X_raw])

    return {
        'X': X,
        'T': treatments,
        'is_interventional': np.asarray(modes),
        'Y_obs': Y_obs,
        'Y0_RR': Y0_RR,
        'Y1_PFCA': Y1_PFCA,
        'p_rr': np.asarray(p_rrs),
    }

def load_and_apply_treatment_policy(
    data_path="./data/scheduling_data.npy",
    T=10,
    confounding_strength=0.5,
    interventional_frac=0.05,
    seed=None,
    save_path=None,
):
    raw_dataset = np.load(data_path, allow_pickle=True).item()
    treated_data = apply_treatment_policy(
        raw_dataset,
        T=T,
        confounding_strength=confounding_strength,
        interventional_frac=interventional_frac,
        seed=seed,
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, treated_data, allow_pickle=True)

    return treated_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/scheduling_data.npy")
    parser.add_argument("--out_path", type=str, default="data/scheduling_treated_data.npy")
    parser.add_argument("--temp", type=float, default=10)
    parser.add_argument("--confounding_strength", type=float, default=1)
    parser.add_argument("--interventional_frac", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    treated_data = load_and_apply_treatment_policy(
        data_path=args.data_path,
        T=args.temp,
        confounding_strength=args.confounding_strength,
        interventional_frac=args.interventional_frac,
        seed=args.seed,
        save_path=args.out_path,
    )

    modes = treated_data['is_interventional']
    treatments = treated_data['T']
    print("Treatment assignment complete.")
    print(f"Observational: {np.sum(modes == 0)} | Interventional: {np.sum(modes == 1)}")
    print(f"RR: {np.sum(treatments == 0)} | PFCA: {np.sum(treatments == 1)}")
