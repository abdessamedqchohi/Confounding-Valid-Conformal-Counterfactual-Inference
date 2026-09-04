import warnings
warnings.filterwarnings('ignore')

import argparse
import json
import os
import sys
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .plotting import plot_results, plot_treatment_realization_intervals
except ImportError:
    from plotting import plot_results, plot_treatment_realization_intervals

def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _fmt_param(value):
    return str(value).replace(".", "p")

def _as_2d_outcome(Y):
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        return Y.reshape(-1, 1)
    return Y

def _outcome_baseline(X, n_outputs):
    X = np.asarray(X, dtype=float)
    if n_outputs == 8 and X.shape[1] >= 8:
        return X[:, :8]
    return np.zeros((len(X), n_outputs), dtype=float)

def _predict_interval_bounds(X, q_lo, q_hi):
    lo = np.asarray(q_lo.predict(X), dtype=float)
    hi = np.asarray(q_hi.predict(X), dtype=float)
    if lo.ndim == 1:
        lo = lo.reshape(-1, 1)
    if hi.ndim == 1:
        hi = hi.reshape(-1, 1)
    baseline = _outcome_baseline(X, lo.shape[1])
    return baseline + lo, baseline + hi

def _score_quantile_from_level(scores, level, allow_infinite=True):
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        raise ValueError("Cannot compute a conformal quantile with no calibration scores.")

    scores_sorted = np.sort(scores)
    level = float(level)
    if level <= 0:
        return scores_sorted[0]

    rank = int(np.ceil(len(scores_sorted) * level))
    if rank > len(scores_sorted):
        return np.inf if allow_infinite else scores_sorted[-1]
    return scores_sorted[max(rank, 1) - 1]

def _weighted_score_quantile_from_level(scores, weights, level, allow_infinite=True):
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(scores) == 0:
        raise ValueError("Cannot compute a weighted quantile with no calibration scores.")
    if len(scores) != len(weights):
        raise ValueError("scores and weights must have the same length.")

    order = np.argsort(scores)
    scores_sorted = scores[order]
    weights_sorted = weights[order]
    weight_sum = np.sum(weights_sorted)
    if weight_sum <= 0:
        raise ValueError("Weighted quantile requires positive total weight.")

    level = float(level)
    if level <= 0:
        return scores_sorted[0]
    if level > 1:
        return np.inf if allow_infinite else scores_sorted[-1]

    cumulative_weight = np.cumsum(weights_sorted) / weight_sum
    idx = np.searchsorted(cumulative_weight, level, side="left")
    return scores_sorted[min(idx, len(scores_sorted) - 1)]

def _conformal_quantile(scores, coverage, allow_infinite=True):
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        raise ValueError("Cannot compute a conformal quantile with no calibration scores.")
    level = float(coverage) * (1.0 + 1.0 / len(scores))
    return _score_quantile_from_level(scores, level, allow_infinite=allow_infinite)

def _weighted_conformal_quantile(scores, weights, coverage, allow_infinite=True):
    scores = np.asarray(scores, dtype=float)
    if len(scores) == 0:
        raise ValueError("Cannot compute a weighted conformal quantile with no calibration scores.")
    level = float(coverage) * (1.0 + 1.0 / len(scores))
    return _weighted_score_quantile_from_level(
        scores,
        weights,
        level,
        allow_infinite=allow_infinite,
    )

class GESPI:
    def __init__(self, alpha=0.20, epsilon=0.05, q_lo=None, q_hi=None, q_lo_it=None, q_hi_it=None, w=None):
        self.alpha = alpha
        self.epsilon = epsilon
        self.q_lo = q_lo
        self.q_hi = q_hi
        if q_lo_it is None:
            self.q_lo_it = q_lo
        else:
            self.q_lo_it = q_lo_it
        if q_hi_it is None:
            self.q_hi_it = q_hi
        else:
            self.q_hi_it = q_hi_it
        self.w_pred = w
        self.q_base, self.q_guard, self.q_synth = None, None, None

    def fit(self, X_obs, Y_obs, X_int, Y_int, target):
        Y_obs = _as_2d_outcome(Y_obs)
        Y_int = _as_2d_outcome(Y_int)
        Y_hat_obs_low, Y_hat_obs_high = _predict_interval_bounds(X_obs, self.q_lo, self.q_hi)
        Y_hat_int_low, Y_hat_int_high = _predict_interval_bounds(X_int, self.q_lo, self.q_hi)
        score_obs = np.max(np.maximum(Y_hat_obs_low- Y_obs, Y_obs - Y_hat_obs_high), axis=1)
        score_int = np.max(np.maximum(Y_hat_int_low- Y_int, Y_int - Y_hat_int_high), axis=1)
        self.q_base = _conformal_quantile(score_int, 1 - self.alpha)
        self.q_guard = _conformal_quantile(score_int, 1 - self.alpha - self.epsilon)
        score_aggr=np.concatenate((score_obs, score_int))
        if target == 0:
            w_obs = 1. / self.w_pred .predict_proba(X_obs)[:, 0]
            w_int = np.ones(len(X_int))/0.5
            w_aggr = np.concatenate((w_obs, w_int))
        else:
            w_obs = 1. / self.w_pred .predict_proba(X_obs)[:, 1]
            w_int = np.ones(len(X_int))/0.5
            w_aggr = np.concatenate((w_obs, w_int))
        self.q_synth = _weighted_conformal_quantile(score_aggr, w_aggr, 1 - self.alpha)
        return self

    def predict(self, X_test):
        L1, U1, L2, U2 = self.predict_set(X_test)
        L_final = np.minimum(L1, np.where(np.isnan(L2), L1, L2))
        U_final = np.maximum(U1, np.where(np.isnan(U2), U1, U2))
        return L_final, U_final

    def predict_set(self, X_test):
        Y_hat_low, Y_hat_high = _predict_interval_bounds(X_test, self.q_lo, self.q_hi)
        Y_hat_low_it, Y_hat_high_it = _predict_interval_bounds(X_test, self.q_lo_it, self.q_hi_it)

        L_b, U_b = Y_hat_low_it - self.q_base, Y_hat_high_it + self.q_base
        L_g, U_g = Y_hat_low_it - self.q_guard, Y_hat_high_it + self.q_guard
        L_m, U_m = Y_hat_low - self.q_synth, Y_hat_high + self.q_synth

        l_ab, u_ab = np.maximum(L_b, L_m), np.minimum(U_b, U_m)
        l_ac, u_ac = np.maximum(L_b, L_g), np.minimum(U_b, U_g)
        valid_ab, valid_ac = l_ab <= u_ab, l_ac <= u_ac

        L1, U1 = np.where(valid_ab, l_ab, np.nan), np.where(valid_ab, u_ab, np.nan)
        L2, U2 = np.where(valid_ac, l_ac, np.nan), np.where(valid_ac, u_ac, np.nan)
        return L1, U1, L2, U2

class CCKE:
    def __init__(self, alpha=None, q_lo=None, q_hi=None, w=None):
        self.alpha = alpha
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.w_pred = w
        self.q_ccke = None

    def fit(self, X_obs, Y_obs, target):
        Y_obs = _as_2d_outcome(Y_obs)
        Y_hat_obs_low, Y_hat_obs_high = _predict_interval_bounds(X_obs, self.q_lo, self.q_hi)
        score_obs = np.max(np.maximum(Y_hat_obs_low - Y_obs, Y_obs - Y_hat_obs_high), axis=1)
        if target == 0:
            w_obs = 1. / self.w_pred.predict_proba(X_obs)[:, 0]
        else:
            w_obs = 1. / self.w_pred.predict_proba(X_obs)[:, 1]
        self.q_ccke = _weighted_conformal_quantile(score_obs, w_obs, 1 - self.alpha)
        return self

    def predict(self, X_test):
        return self.predict_set(X_test)

    def predict_set(self, X_test):
        Y_hat_low, Y_hat_high = _predict_interval_bounds(X_test, self.q_lo, self.q_hi)
        L, U = Y_hat_low - self.q_ccke, Y_hat_high + self.q_ccke
        return L, U

class NCCKE:
    def __init__(self, alpha=0.20, q_lo=None, q_hi=None):
        self.alpha = alpha
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.q_nccke = None

    def fit(self, X_obs, Y_obs, target):
        Y_obs = _as_2d_outcome(Y_obs)
        Y_hat_obs_low, Y_hat_obs_high = _predict_interval_bounds(X_obs, self.q_lo, self.q_hi)
        score_obs = np.max(np.maximum(Y_hat_obs_low - Y_obs, Y_obs - Y_hat_obs_high), axis=1)
        self.q_nccke = _conformal_quantile(score_obs, 1 - self.alpha)
        return self

    def predict(self, X_test):
        return self.predict_set(X_test)

    def predict_set(self, X_test):
        Y_hat_low, Y_hat_high = _predict_interval_bounds(X_test, self.q_lo, self.q_hi)
        L, U = Y_hat_low - self.q_nccke, Y_hat_high + self.q_nccke
        return L, U

class NaiveCP:
    def __init__(self, alpha=0.20, q_lo=None, q_hi=None):
        self.alpha = alpha
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.q_naive = None

    def fit(self, X_int, Y_int, target):
        if len(X_int) == 0:
            raise ValueError("Naive CP requires interventional calibration data for the requested treatment.")
        Y_int = _as_2d_outcome(Y_int)
        Y_hat_int_low, Y_hat_int_high = _predict_interval_bounds(X_int, self.q_lo, self.q_hi)
        scores = np.max(np.maximum(Y_hat_int_low - Y_int, Y_int - Y_hat_int_high), axis=1)
        self.q_naive = _split_conformal_quantile(scores, self.alpha)
        return self

    def predict(self, X_test):
        return self.predict_set(X_test)

    def predict_set(self, X_test):
        Y_hat_low, Y_hat_high = _predict_interval_bounds(X_test, self.q_lo, self.q_hi)
        return Y_hat_low - self.q_naive, Y_hat_high + self.q_naive

class CKE:
    def __init__(self, q_lo=None, q_hi=None):
        self.q_lo = q_lo
        self.q_hi = q_hi

    def predict(self, X_test):
        return self.predict_set(X_test)

    def predict_set(self, X_test):
        Y_hat_low, Y_hat_high = _predict_interval_bounds(X_test, self.q_lo, self.q_hi)
        return Y_hat_low, Y_hat_high

class ClassifierDensityRatio:
    def __init__(self, D_inter, D_obs, seed=42, clip=0.05):
        self.clip = clip
        X = np.concatenate([D_obs, D_inter], axis=0)
        y = np.concatenate([np.ones(len(D_obs)), np.zeros(len(D_inter))])
        self.clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=10000, C=100.0, random_state=seed),
        ).fit(X, y)

    def compute_density_ratio(self, D):
        proba = self.clf.predict_proba(D)
        obs_col = int(np.where(self.clf.classes_ == 1)[0][0])
        int_col = int(np.where(self.clf.classes_ == 0)[0][0])
        p_obs = np.clip(proba[:, obs_col], self.clip, 1 - self.clip)
        p_int = np.clip(proba[:, int_col], self.clip, 1 - self.clip)
        return p_int / p_obs

def weighted_conformal(alpha, weights_calib, weights_test, scores, allow_infinite=False):
    scores = np.asarray(scores, dtype=float)
    w_calib = np.asarray(weights_calib, dtype=float)
    scalar_test_weight = np.ndim(weights_test) == 0
    w_test = np.atleast_1d(np.asarray(weights_test, dtype=float))
    weight_sum = np.sum(w_calib)
    if weight_sum <= 0:
        raise ValueError("Weighted conformal requires positive total calibration weight.")

    levels = (1 - alpha) * (weight_sum + w_test) / weight_sum
    q = np.asarray([
        _weighted_score_quantile_from_level(
            scores,
            w_calib,
            level,
            allow_infinite=allow_infinite,
        )
        for level in levels
    ])
    return q[0] if scalar_test_weight else q

def _split_conformal_quantile(scores, alpha):
    return _conformal_quantile(scores, 1 - alpha)

def _xy_features(X, Y):
    return np.concatenate([np.asarray(X, dtype=float), _as_2d_outcome(Y)], axis=1)

def _fit_endpoint_regressor(X, Y):
    model =  MultiOutputRegressor(
        make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(),
        ))
    return model.fit(X, Y)

def _first_stage_wscp_intervals(X_obs, Y_obs, X_int, Y_int, alpha,q_lo,q_hi, seed=42):
    if len(X_obs) == 0:
        raise ValueError("WSCP requires observational calibration data for the requested treatment.")
    if len(X_int) == 0:
        raise ValueError("WSCP requires interventional calibration data for the requested treatment.")

    Y_obs = _as_2d_outcome(Y_obs)
    Y_int = _as_2d_outcome(Y_int)
    D_obs = _xy_features(X_obs, Y_obs)
    D_int = _xy_features(X_int, Y_int)
    density_ratio = ClassifierDensityRatio(D_inter=D_int, D_obs=D_obs, seed=seed)
    w_obs = density_ratio.compute_density_ratio(D_obs)
    w_int = density_ratio.compute_density_ratio(D_int)
    C_L = np.empty_like(Y_int, dtype=float)
    C_R = np.empty_like(Y_int, dtype=float)
    Y_hat_obs_low, Y_hat_obs_high = _predict_interval_bounds(X_obs, q_lo, q_hi)
    score_obs = np.max(np.maximum(Y_hat_obs_low - Y_obs, Y_obs - Y_hat_obs_high), axis=1)
    for j in range(len(X_int)):
        
        q_F = weighted_conformal(alpha, w_obs, w_int[j], score_obs, allow_infinite=False)
        Y_hat_int_low, Y_hat_int_high = _predict_interval_bounds(X_int[j:j + 1], q_lo, q_hi)
        C_L[j] = Y_hat_int_low - q_F
        C_R[j] = Y_hat_int_high + q_F
    return C_L, C_R

class WSCP_Inexact:
    def __init__(self, alpha=0.20, q_lo=None, q_hi=None, w_pred=None, seed=42):
        self.alpha = alpha
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.w_pred = w_pred
        self.seed = seed
        self.lower_model = None
        self.upper_model = None

    def fit(self, X_obs_all, T_obs_all, Y_obs_all, X_int_all, T_int_all, Y_int_all, target):
        X_obs = X_obs_all[T_obs_all == target]
        Y_obs = Y_obs_all[T_obs_all == target]
        X_int = X_int_all[T_int_all == target]
        Y_int = Y_int_all[T_int_all == target]
        C_L, C_R = _first_stage_wscp_intervals(X_obs, Y_obs, X_int, Y_int, self.alpha,self.q_lo,self.q_hi, seed=self.seed)
        baseline = _outcome_baseline(X_int, C_L.shape[1])
        self.lower_model = _fit_endpoint_regressor(X_int, C_L - baseline)
        self.upper_model = _fit_endpoint_regressor(X_int, C_R - baseline)
        return self

    def predict_set(self, X_test):
        pred_low = np.asarray(self.lower_model.predict(X_test), dtype=float)
        pred_high = np.asarray(self.upper_model.predict(X_test), dtype=float)
        baseline = _outcome_baseline(X_test, pred_low.shape[1])
        return baseline + pred_low, baseline + pred_high

class WSCP_Exact:
    def __init__(self, alpha=0.20, q_lo=None, q_hi=None, w_pred=None, seed=42):
        self.alpha = alpha
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.w_pred = w_pred
        self.seed = seed
        self.lower_model = None
        self.upper_model = None
        self.q_F = 0.0

    def fit(self, X_obs_all, T_obs_all, Y_obs_all, X_int_all, T_int_all, Y_int_all, target):
        X_obs = X_obs_all[T_obs_all == target]
        Y_obs = Y_obs_all[T_obs_all == target]
        X_int = X_int_all[T_int_all == target]
        Y_int = Y_int_all[T_int_all == target]
        C_L, C_R = _first_stage_wscp_intervals(X_obs, Y_obs, X_int, Y_int, self.alpha,self.q_lo,self.q_hi, seed=self.seed)

        n_train = len(X_int) // 2
        if n_train == 0 or n_train == len(X_int):
            raise ValueError("WSCP exact requires at least two interventional samples for the requested treatment.")
        X_int_tr, X_int_cal = X_int[:n_train], X_int[n_train:]
        C_L_tr, C_L_cal = C_L[:n_train], C_L[n_train:]
        C_R_tr, C_R_cal = C_R[:n_train], C_R[n_train:]
        baseline_tr = _outcome_baseline(X_int_tr, C_L_tr.shape[1])
        self.lower_model = _fit_endpoint_regressor(X_int_tr, C_L_tr - baseline_tr)
        self.upper_model = _fit_endpoint_regressor(X_int_tr, C_R_tr - baseline_tr)

        pred_L_cal = np.asarray(self.lower_model.predict(X_int_cal), dtype=float)
        pred_R_cal = np.asarray(self.upper_model.predict(X_int_cal), dtype=float)
        baseline_cal = _outcome_baseline(X_int_cal, pred_L_cal.shape[1])
        pred_L_cal = pred_L_cal + baseline_cal
        pred_R_cal = pred_R_cal + baseline_cal
        scores = np.max(np.maximum(pred_L_cal - C_L_cal, C_R_cal - pred_R_cal), axis=1)
        self.q_F = _split_conformal_quantile(scores, self.alpha)
        return self

    def predict_set(self, X_test):
        pred_low = np.asarray(self.lower_model.predict(X_test), dtype=float)
        pred_high = np.asarray(self.upper_model.predict(X_test), dtype=float)
        baseline = _outcome_baseline(X_test, pred_low.shape[1])
        L = baseline + pred_low - self.q_F
        R = baseline + pred_high + self.q_F
        return L, R

def evaluate(C, Y_0, Y_1):
    def _as_intervals(pred_set):
        if len(pred_set) == 2:
            L, U = pred_set
            return [(np.asarray(L, dtype=float), np.asarray(U, dtype=float))]
        if len(pred_set) == 4:
            L1, U1, L2, U2 = pred_set
            return [
                (np.asarray(L1, dtype=float), np.asarray(U1, dtype=float)),
                (np.asarray(L2, dtype=float), np.asarray(U2, dtype=float)),
            ]
        raise ValueError("Prediction sets must be (L, U) or (L1, U1, L2, U2).")

    def _coverage(y, intervals):
        covered = np.zeros_like(y, dtype=bool)
        for L, U in intervals:
            covered |= (L <= y) & (y <= U)
        if covered.ndim == 1:
            return float(np.mean(covered))
        return float(np.mean(np.all(covered, axis=1)))

    def _set_width(intervals):
        if len(intervals) == 1:
            L, U = intervals[0]
            return float(np.nanmean(np.maximum(U - L, 0.0)))

        L1, U1 = intervals[0]
        L2, U2 = intervals[1]
        w1 = np.maximum(U1 - L1, 0.0)
        w2 = np.maximum(U2 - L2, 0.0)
        overlap = np.maximum(np.minimum(U1, U2) - np.maximum(L1, L2), 0.0)
        return float(np.nanmean(w1 + w2 - overlap))

    def _hull(intervals):
        lowers = np.stack([L for L, _ in intervals], axis=0)
        uppers = np.stack([U for _, U in intervals], axis=0)
        return np.nanmin(lowers, axis=0), np.nanmax(uppers, axis=0)

    Y_0 = _as_2d_outcome(Y_0)
    Y_1 = _as_2d_outcome(Y_1)
    ite = Y_1 - Y_0

    C_0 = _as_intervals(C['C_0'])
    C_1 = _as_intervals(C['C_1'])

    L_0, U_0 = _hull(C_0)
    L_1, U_1 = _hull(C_1)
    C_ite = [(L_1 - U_0, U_1 - L_0)]

    performance = {
        'cov_0': _coverage(Y_0, C_0),
        'cov_1': _coverage(Y_1, C_1),
        'cov_ite': _coverage(ite, C_ite),
        'width_0': _set_width(C_0),
        'width_1': _set_width(C_1),
        'width_ite': _set_width(C_ite),
    }
    return performance

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

def _make_folds(n_samples, n_folds, seed=0, shuffle=True):
    if n_folds < 1:
        raise ValueError("n_folds must be at least 1.")
    if n_folds > n_samples:
        raise ValueError("n_folds cannot be larger than the number of samples.")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    return np.array_split(indices, n_folds)

def _summarize_fold_logs(fold_logs):
    method_names = fold_logs[0]['methods'].keys()
    summary = {}
    for method in method_names:
        metric_names = fold_logs[0]['methods'][method].keys()
        summary[method] = {}
        for metric in metric_names:
            values = np.asarray([fold['methods'][method][metric] for fold in fold_logs], dtype=float)
            summary[method][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
    return summary

def _run_experiment_fold(
    dataset_obs_cal,
    dataset_int_cal,
    dataset_te,
    predictors,
    dataset_obs_wscp=None,
    alpha=0.2,
    epsilon=0.05,
):
    C_GESPI={'C_0': None,'C_1': None}
    C_CCKE = {'C_0': None, 'C_1': None}
    C_NCCKE = {'C_0': None, 'C_1': None}
    C_Naive = {'C_0': None, 'C_1': None}
    C_CKE = {'C_0': None, 'C_1': None}
    C_WSCP_Inexact = {'C_0': None, 'C_1': None}
    C_WSCP_Exact = {'C_0': None, 'C_1': None}

    dataset_obs_wscp = dataset_obs_cal if dataset_obs_wscp is None else dataset_obs_wscp

    X_obs_all = dataset_obs_wscp['X'].astype('float')
    T_obs_all = dataset_obs_wscp['T'].astype(int)
    Y_obs_all = dataset_obs_wscp['Y_obs'].astype('float')
    X_te = dataset_te['X'].astype('float')

    for target in [0,1]:
        print('Running Experiments for Target: ' + str(target)+ ' on interventional data size: '+str(len(dataset_int_cal['X']))+' and observational data size: '+str(len(dataset_obs_cal['X'])))
        id_tgt_obs = dataset_obs_cal['T']==target
        X_obs = dataset_obs_cal['X'][id_tgt_obs].astype('float')
        Y_obs = dataset_obs_cal['Y_obs'][id_tgt_obs].astype('float')

        id_tgt_int = dataset_int_cal['T'] == target
        X_int = dataset_int_cal['X'][id_tgt_int].astype('float')
        Y_int = dataset_int_cal['Y_obs'][id_tgt_int].astype('float')
        X_int_all = dataset_int_cal['X']
        Y_int_all = dataset_int_cal['Y_obs']
        
        gespi = GESPI(alpha=alpha, epsilon=epsilon,q_lo=predictors[f'qr_{target}_lo'],q_hi=predictors[f'qr_{target}_hi'],q_lo_it=None,q_hi_it=None,w=predictors['pr_score_est'])
        gespi.fit(X_obs, Y_obs, X_int, Y_int, target)
        C_GESPI[f'C_{target}']=gespi.predict_set(X_te)
        
        ccke= CCKE(alpha=alpha, q_lo=predictors[f'qr_{target}_lo'],q_hi=predictors[f'qr_{target}_hi'],w=predictors['pr_score_est'])
        ccke.fit(X_obs, Y_obs, target)
        C_CCKE[f'C_{target}']=ccke.predict_set(X_te)
        
        nccke= NCCKE(alpha=alpha, q_lo=predictors[f'qr_{target}_lo'],q_hi=predictors[f'qr_{target}_hi'])
        nccke.fit(X_obs, Y_obs, target)
        C_NCCKE[f'C_{target}']=nccke.predict_set(X_te)
        
        naive = NaiveCP(alpha=alpha+epsilon, q_lo=predictors[f'qr_{target}_lo'], q_hi=predictors[f'qr_{target}_hi'])
        naive.fit(X_int, Y_int, target)
        C_Naive[f'C_{target}'] = naive.predict_set(X_te)
        
        cke= CKE(q_lo=predictors[f'qr_{target}_lo'],q_hi=predictors[f'qr_{target}_hi'])
        C_CKE[f'C_{target}']=cke.predict_set(X_te)

        T_int=dataset_int_cal['T'].astype(int)
        wscp_in = WSCP_Inexact(alpha=alpha, q_lo=predictors[f'qr_{target}_lo'], q_hi=predictors[f'qr_{target}_hi'], w_pred=predictors['pr_score_est'])
        wscp_in.fit(X_obs_all, T_obs_all, Y_obs_all, X_int_all, T_int, Y_int_all, target)
        C_WSCP_Inexact[f'C_{target}'] = wscp_in.predict_set(X_te)

        wscp_ex = WSCP_Exact(alpha=alpha, q_lo=predictors[f'qr_{target}_lo'], q_hi=predictors[f'qr_{target}_hi'], w_pred=predictors['pr_score_est'])
        wscp_ex.fit(X_obs_all, T_obs_all, Y_obs_all, X_int_all, T_int, Y_int_all, target)
        C_WSCP_Exact[f'C_{target}'] = wscp_ex.predict_set(X_te)

    Y_0_te = dataset_te['Y0'].astype('float')
    Y_1_te = dataset_te['Y1'].astype('float')
    GESPI_log=evaluate(C_GESPI,Y_0_te,Y_1_te)
    CCKE_log=evaluate(C_CCKE,Y_0_te,Y_1_te)
    NCCKE_log=evaluate(C_NCCKE,Y_0_te,Y_1_te)
    Naive_log=evaluate(C_Naive,Y_0_te,Y_1_te)
    CKE_log=evaluate(C_CKE,Y_0_te,Y_1_te)
    WSCP_Inexact_log=evaluate(C_WSCP_Inexact,Y_0_te,Y_1_te)
    WSCP_Exact_log=evaluate(C_WSCP_Exact,Y_0_te,Y_1_te)
    return GESPI_log, CCKE_log, NCCKE_log, Naive_log, CKE_log, WSCP_Inexact_log, WSCP_Exact_log

def run_experiment(
    dataset_obs,
    dataset_int,
    dataset_te,
    predictors,
    dataset_obs_train=None,
    n_folds=5,
    n_obs=2000,
    n_int=50,
    n_test=1000,
    alpha=0.2,
    epsilon=0.05,
    seed=0,
):
    n_obs = min(int(n_obs), len(dataset_obs['X']))
    n_int = min(int(n_int), len(dataset_int['X']))
    n_test = min(int(n_test), len(dataset_te['X']))
    if n_obs < 1 or n_int < 2 or n_test < 1:
        raise ValueError("Each fold needs at least one observational sample, two interventional samples, and one test sample.")

    rng = np.random.default_rng(seed)
    print('data per fold - n_test: ' + str(n_test) + '  n_int: ' + str(n_int) + '  n_obs: ' + str(n_obs))
    fold_logs = []
    for fold_id in range(n_folds):
        print( 'Fold :'+str(fold_id))
        obs_idx=rng.choice(len(dataset_obs['X']), size=n_obs, replace=False)
        idx_0 = np.where(dataset_int['T'] == 0)[0]
        idx_1 = np.where(dataset_int['T'] == 1)[0]
        c0 = rng.choice(idx_0, size=min(len(idx_0), n_int // 2), replace=False)
        c1 = rng.choice(idx_1, size=min(len(idx_1), n_int - len(c0)), replace=False)
        int_idx = np.concatenate([c0, c1])
        te_idx=rng.choice(len(dataset_te['X']), size=n_test, replace=False)
        dataset_obs_cal_fold = _slice_dataset(dataset_obs, obs_idx)
        dataset_int_fold = _slice_dataset(dataset_int, int_idx)
        dataset_test_fold = _slice_dataset(dataset_te, te_idx)
        dataset_obs_wscp = (
            _concat_datasets(dataset_obs_train, dataset_obs_cal_fold)
            if dataset_obs_train is not None
            else dataset_obs_cal_fold
        )
        GESPI_log, CCKE_log, NCCKE_log, Naive_log, CKE_log, WSCP_Inexact_log, WSCP_Exact_log = _run_experiment_fold(
            dataset_obs_cal_fold,
            dataset_int_fold,
            dataset_test_fold,
            predictors,
            dataset_obs_wscp=dataset_obs_wscp,
            alpha=alpha,
            epsilon=epsilon,
        )
        fold_logs.append({
            'fold': fold_id,
            'methods': {
                'GESPI': GESPI_log,
                'CCKE': CCKE_log,
                'NCCKE': NCCKE_log,
                'Naive': Naive_log,
                'CKE': CKE_log,
                'WSCP_Inexact': WSCP_Inexact_log,
                'WSCP_Exact': WSCP_Exact_log,
            },
        })

    return {
        'folds': fold_logs,
        'summary': _summarize_fold_logs(fold_logs),
        'alpha': alpha,
        'epsilon': epsilon,
        'n_obs_per_fold': n_obs,
        'n_int_per_fold': n_int,
        'n_test_per_fold': n_test,
    }

def results_path(results_dir=None, alpha=0.2, epsilon=0.05, confounding_strength=None):
    if results_dir is None:
        results_dir = os.path.join(_repo_root(), "results")
    tag = f"alpha{_fmt_param(alpha)}_eps{_fmt_param(epsilon)}"
    if confounding_strength is not None:
        tag += f"_conf{_fmt_param(confounding_strength)}"
    return os.path.join(results_dir, f"results_{tag}.json")

def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")

def save_results(results, results_dir=None, confounding_strength=None):
    if results_dir is None:
        results_dir = os.path.join(_repo_root(), "results")
    os.makedirs(results_dir, exist_ok=True)
    payload = dict(results)
    if confounding_strength is not None:
        payload["confounding_strength"] = confounding_strength
    path = results_path(
        results_dir=results_dir,
        alpha=payload.get("alpha", 0.2),
        epsilon=payload.get("epsilon", 0.05),
        confounding_strength=confounding_strength,
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    return path

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def fit_base_predictors(X,Y,alpha):
    X = np.asarray(X, dtype=float)
    Y = _as_2d_outcome(Y)
    residual = Y - _outcome_baseline(X, Y.shape[1])
    base_predictor_lo = MultiOutputRegressor(
        make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor( loss="quantile", alpha=(alpha / 2)),
        )
    )
    base_predictor_hi = MultiOutputRegressor(
        make_pipeline(
            StandardScaler(),
            GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2),
        )
    )
    base_predictor_lo.fit(X, residual)
    base_predictor_hi.fit(X, residual)
    return base_predictor_lo, base_predictor_hi

def fit_likelihood_ratio_estimator(X,T):
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=10000, C=100.0),
    )
    clf.fit(X, T)
    return clf

def load_handover_dataset(data_path):
    dataset = np.load(data_path, allow_pickle=True).item()
    required = ["X", "T", "is_interventional", "Y_obs", "Y0", "Y1"]
    missing = [key for key in required if key not in dataset]
    if missing:
        raise KeyError(f"{data_path} is missing keys: {', '.join(missing)}")
    return {key: np.asarray(value) for key, value in dataset.items()}

def _load_handover_generator():
    root = _repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)
    import generate_handover_dataset
    return generate_handover_dataset

def _parse_float_list(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values

def _validate_confounding_strengths(values):
    for value in values:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Confounding strength must be in [0, 1]; got {value}.")

def _generated_dataset_path(data_dir, confounding_strength):
    return os.path.join(
        data_dir,
        f"handover_dataset_conf{_fmt_param(confounding_strength)}.npy",
    )

def generate_handover_dataset(args, confounding_strength):
    generator = _load_handover_generator()
    output_file = _generated_dataset_path(args.generated_data_dir, confounding_strength)
    gen_args = argparse.Namespace(
        trace_glob=args.trace_glob,
        output_file=output_file,
        num_samples=args.num_samples,
        int_frac=args.int_frac,
        history_s=args.history_s,
        outcome_horizon_s=args.outcome_horizon_s,
        min_opportunity_steps=args.min_opportunity_steps,
        load_min=args.load_min,
        load_max=args.load_max,
        proposal_margin_db=args.proposal_margin_db,
        proposal_slope=args.proposal_slope,
        accept_load_threshold=args.accept_load_threshold,
        accept_slope=args.accept_slope,
        confounding_strength=confounding_strength,
        min_resource=args.min_resource,
        gain_reference=args.gain_reference,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )
    traces = generator.load_traces(gen_args.trace_glob)
    samples, gain_reference = generator.generate_dataset(traces, gen_args)
    dataset = generator.save_dataset(samples, gain_reference, gen_args, traces)
    treatment_rate = float(np.mean(dataset["T"]))
    print(
        "Generated handover dataset for confounding_strength="
        + str(confounding_strength)
        + " with effective_accept_slope="
        + str(args.accept_slope * confounding_strength)
        + " and treatment_rate="
        + f"{treatment_rate:.3f}"
    )
    print("Saved generated dataset to: " + output_file)
    return {key: np.asarray(value) for key, value in dataset.items()}

def split_observational_interventional(dataset):
    mask_int = np.asarray(dataset["is_interventional"], dtype=int) == 1
    mask_obs = ~mask_int
    dataset_int = {key: np.asarray(value)[mask_int] for key, value in dataset.items()}
    dataset_obs = {key: np.asarray(value)[mask_obs] for key, value in dataset.items()}
    if len(dataset_int["X"]) == 0 or len(dataset_obs["X"]) == 0:
        raise ValueError("Dataset must contain both interventional and observational rows.")
    return dataset_obs, dataset_int

def split_train_test_cal(dataset_obs, n_train, n_test):
    n_available = len(dataset_obs["X"])
    if n_train < 1:
        raise ValueError("--n-train must be at least 1.")
    if n_train + n_test >= n_available:
        raise ValueError(
            f"Need n_train + n_test < observational rows ({n_available}); "
            f"got {n_train} + {n_test}."
        )
    dataset_train = {key: value[:n_train] for key, value in dataset_obs.items()}
    dataset_te = {key: value[n_train:n_train+n_test] for key, value in dataset_obs.items()}
    dataset_cal = {key: value[n_train+n_test:] for key, value in dataset_obs.items()}
    return dataset_train, dataset_te, dataset_cal

def build_predictors(dataset_train, alpha):
    is_t0 = dataset_train['T'] == 0
    is_t1 = dataset_train['T'] == 1
    if not np.any(is_t0) or not np.any(is_t1):
        raise ValueError("Training split must contain both T=0 and T=1 rows.")
    print('Total number of T=0 data : '+str(np.sum(is_t0))+'  --  Total number of T=1 data : '+str(np.sum(is_t1)))
    qr_0_lo, qr_0_hi = fit_base_predictors(dataset_train['X'][is_t0], dataset_train['Y_obs'][is_t0], alpha)
    qr_1_lo, qr_1_hi = fit_base_predictors(dataset_train['X'][is_t1], dataset_train['Y_obs'][is_t1], alpha)
    pr_score_est = fit_likelihood_ratio_estimator(dataset_train['X'], dataset_train['T'])
    return {
        'qr_0_lo': qr_0_lo,
        'qr_0_hi': qr_0_hi,
        'qr_1_lo': qr_1_lo,
        'qr_1_hi': qr_1_hi,
        'pr_score_est': pr_score_est,
    }

def parse_args():
    root = _repo_root()
    parser = argparse.ArgumentParser(description="Run GESPI/CCKE experiments on the handover dataset.")
    parser.add_argument("--data-path", default=os.path.join(root, "data", "handover_dataset.npy"))
    parser.add_argument("--results-dir", default=os.path.join(root, "results"))
    parser.add_argument("--plots-dir", default=os.path.join(root, "plots"))
    parser.add_argument(
        "--generate-datasets",
        default=True,
        action="store_true",
        help="Generate handover datasets from ray-tracing traces instead of loading --data-path.",
    )
    parser.add_argument(
        "--confounding-strengths",
        default="0,0.25,0.5,0.75,1",
        help=(
            "Comma-separated strengths in [0, 1]. Used with --generate-datasets; "
            "each value multiplies --accept-slope in the generator."
        ),
    )
    parser.add_argument(
        "--generated-data-dir",
        default=os.path.join(root, "data", "generated_handover"),
        help="Directory for datasets generated during a confounding sweep.",
    )
    parser.add_argument(
        "--trace-glob",
        default=os.path.join(root, "ray_tracing_data_generation", "traces", "*_rss_trace.npz"),
        help="Trace glob passed to generate_handover_dataset.py.",
    )
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--int-frac", type=float, default=0.05)
    parser.add_argument("--history-s", type=float, default=10.0)
    parser.add_argument("--outcome-horizon-s", type=float, default=10.0)
    parser.add_argument("--min-opportunity-steps", type=int, default=10)
    parser.add_argument("--load-min", type=float, default=0.05)
    parser.add_argument("--load-max", type=float, default=0.95)
    parser.add_argument("--proposal-margin-db", type=float, default=5.0)
    parser.add_argument("--proposal-slope", type=float, default=0.8)
    parser.add_argument("--accept-load-threshold", type=float, default=0.65)
    parser.add_argument("--accept-slope", type=float, default=8.0)
    parser.add_argument("--min-resource", type=float, default=0.05)
    parser.add_argument("--gain-reference", default="auto")
    parser.add_argument("--max-attempts", type=int, default=200000)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=0.025)
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-test-split", type=int, default=2000)
    parser.add_argument("--n-folds", type=int, default=100)
    parser.add_argument("--n-obs-per-fold", type=int, default=2000)
    parser.add_argument("--n-int-per-fold", type=int, default=50)
    parser.add_argument("--n-test-per-fold", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-interval-plots", action="store_true")
    args = parser.parse_args()
    if args.generate_datasets:
        args.confounding_strengths = _parse_float_list(args.confounding_strengths)
        _validate_confounding_strengths(args.confounding_strengths)
    else:
        args.confounding_strengths = [None]
    return args

def main():
    args = parse_args()
    for confounding_strength in args.confounding_strengths:
        if args.generate_datasets:
            print("Running generated handover experiment with confounding_strength=" + str(confounding_strength))
            dataset = generate_handover_dataset(args, confounding_strength)
        else:
            print("Running handover experiment from: " + args.data_path)
            dataset = load_handover_dataset(args.data_path)

        dataset_obs, dataset_int = split_observational_interventional(dataset)
        dataset_train, dataset_te, dataset_cal = split_train_test_cal(
            dataset_obs,
            args.n_train,
            args.n_test_split,
        )
        predictors = build_predictors(dataset_train, args.alpha)
        results = run_experiment(
            dataset_cal,
            dataset_int,
            dataset_te,
            predictors,
            dataset_obs_train=dataset_train,
            n_folds=args.n_folds,
            n_obs=args.n_obs_per_fold,
            n_int=args.n_int_per_fold,
            n_test=args.n_test_per_fold,
            alpha=args.alpha,
            epsilon=args.epsilon,
            seed=args.seed,
        )
        saved_results_path = save_results(
            results,
            results_dir=args.results_dir,
            confounding_strength=confounding_strength,
        )
        print("Saved results to: " + saved_results_path)
        plot_results(
            results,
            out_path_prefix=os.path.join(args.plots_dir, "results"),
            alpha=args.alpha,
            epsilon=args.epsilon,
            confounding_strength=confounding_strength,
        )

        if not args.skip_interval_plots:
            for plot_target in [0, 1, 'ITE']:
                plot_treatment_realization_intervals(
                    dataset_cal,
                    dataset_int,
                    dataset_te,
                    predictors,
                    {'GESPI': GESPI, 'CCKE': CCKE, 'NCCKE': NCCKE, 'Naive': NaiveCP, 'CKE': CKE, 'WSCP_Inexact': WSCP_Inexact, 'WSCP_Exact': WSCP_Exact},
                    dataset_obs_train=dataset_train,
                    n=10,
                    target=plot_target,
                    out_dir=os.path.join(args.plots_dir, f"treatment_{plot_target}_intervals"),
                    alpha=args.alpha,
                    epsilon=args.epsilon,
                    confounding_strength=confounding_strength,
                )

if __name__ == '__main__':
    main()
