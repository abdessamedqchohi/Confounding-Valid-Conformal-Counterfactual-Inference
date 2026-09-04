import warnings
warnings.filterwarnings('ignore')

import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .plotting import plot_results, plot_treatment_realization_intervals
except ImportError:
    from plotting import plot_results, plot_treatment_realization_intervals

try:
    from policy_treatment import load_and_apply_treatment_policy
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from policy_treatment import load_and_apply_treatment_policy

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
        
        Y_hat_obs_low, Y_hat_obs_high = X_obs[:, :8] + self.q_lo.predict(X_obs), X_obs[:, :8] + self.q_hi.predict(X_obs)
        Y_hat_int_low, Y_hat_int_high = X_int[:, :8] +  self.q_lo.predict(X_int), X_int[:, :8] + self.q_hi.predict(X_int)
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
        Y_hat_low, Y_hat_high = X_test[:, :8] + self.q_lo.predict(X_test), X_test[:, :8] + self.q_hi.predict(X_test)
        Y_hat_low_it, Y_hat_high_it = X_test[:, :8] + self.q_lo.predict(X_test), X_test[:, :8] + self.q_hi.predict(X_test)

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
        Y_hat_obs_low, Y_hat_obs_high = X_obs[:, :8] + self.q_lo.predict(X_obs), X_obs[:, :8] + self.q_hi.predict(X_obs)
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
        Y_hat_low, Y_hat_high = X_test[:, :8] + self.q_lo.predict(X_test), X_test[:, :8] + self.q_hi.predict(X_test)
        L, U = Y_hat_low - self.q_ccke, Y_hat_high + self.q_ccke
        return L, U

class NCCKE:

    def __init__(self, alpha=0.20, q_lo=None, q_hi=None):
        self.alpha = alpha
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.q_nccke = None

    def fit(self, X_obs, Y_obs, target):
        Y_hat_obs_low, Y_hat_obs_high = X_obs[:, :8] + self.q_lo.predict(X_obs), X_obs[:, :8] + self.q_hi.predict(X_obs)
        score_obs = np.max(np.maximum(Y_hat_obs_low - Y_obs, Y_obs - Y_hat_obs_high), axis=1)
        self.q_nccke = _conformal_quantile(score_obs, 1 - self.alpha)
        return self

    def predict(self, X_test):
        return self.predict_set(X_test)

    def predict_set(self, X_test):
        Y_hat_low, Y_hat_high = X_test[:, :8] + self.q_lo.predict(X_test), X_test[:, :8] + self.q_hi.predict(X_test)
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
        Y_hat_int_low = X_int[:, :8] + self.q_lo.predict(X_int)
        Y_hat_int_high = X_int[:, :8] + self.q_hi.predict(X_int)
        scores = np.max(np.maximum(Y_hat_int_low - Y_int, Y_int - Y_hat_int_high), axis=1)
        self.q_naive = _split_conformal_quantile(scores, self.alpha)
        return self

    def predict(self, X_test):
        return self.predict_set(X_test)

    def predict_set(self, X_test):
        Y_hat_low = X_test[:, :8] + self.q_lo.predict(X_test)
        Y_hat_high = X_test[:, :8] + self.q_hi.predict(X_test)
        return Y_hat_low - self.q_naive, Y_hat_high + self.q_naive

class CKE:

    def __init__(self, q_lo=None, q_hi=None):
        self.q_lo = q_lo
        self.q_hi = q_hi

    def predict(self, X_test):
        return self.predict_set(X_test)

    def predict_set(self, X_test):
        Y_hat_low, Y_hat_high = X_test[:, :8] + self.q_lo.predict(X_test), X_test[:, :8] + self.q_hi.predict(X_test)
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
    return np.concatenate([np.asarray(X, dtype=float), np.asarray(Y, dtype=float)], axis=1)

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

    D_obs = _xy_features(X_obs, Y_obs)
    D_int = _xy_features(X_int, Y_int)
    density_ratio = ClassifierDensityRatio(D_inter=D_int, D_obs=D_obs, seed=seed)
    w_obs = density_ratio.compute_density_ratio(D_obs)
    w_int = density_ratio.compute_density_ratio(D_int)
    C_L = np.empty_like(Y_int, dtype=float)
    C_R = np.empty_like(Y_int, dtype=float)
    Y_hat_obs_low, Y_hat_obs_high = X_obs[:, :8] + q_lo.predict(X_obs), X_obs[:, :8] + q_hi.predict(X_obs)
    score_obs = np.max(np.maximum(Y_hat_obs_low - Y_obs, Y_obs - Y_hat_obs_high), axis=1)
    for j in range(len(X_int)):
        
        q_F = weighted_conformal(alpha, w_obs, w_int[j], score_obs, allow_infinite=False)
        C_L[j] = X_int[j:j + 1,:8]+ q_lo.predict(X_int[j:j + 1]) - q_F
        C_R[j] = X_int[j:j + 1,:8]+ q_hi.predict(X_int[j:j + 1]) + q_F
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
        self.lower_model = _fit_endpoint_regressor(X_int, C_L-X_int[:,:8])
        self.upper_model = _fit_endpoint_regressor(X_int, C_R-X_int[:,:8])
        return self

    def predict_set(self, X_test):
        return X_test[:, :8]+self.lower_model.predict(X_test), X_test[:, :8]+self.upper_model.predict(X_test)

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
        self.lower_model = _fit_endpoint_regressor(X_int_tr, C_L_tr-X_int_tr[:,:8])
        self.upper_model = _fit_endpoint_regressor(X_int_tr, C_R_tr-X_int_tr[:,:8])

        pred_L_cal =self.lower_model.predict(X_int_cal)+X_int_cal[:,:8]
        pred_R_cal = self.upper_model.predict(X_int_cal)+X_int_cal[:,:8]
        scores = np.max(np.maximum(pred_L_cal - C_L_cal, C_R_cal - pred_R_cal), axis=1)
        self.q_F = _split_conformal_quantile(scores, self.alpha)
        return self

    def predict_set(self, X_test):
        L =X_test[:,:8]+(self.lower_model.predict(X_test) - self.q_F)
        R =X_test[:,:8]+ (self.upper_model.predict(X_test) + self.q_F)
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

    Y_0 = np.asarray(Y_0, dtype=float)
    Y_1 = np.asarray(Y_1, dtype=float)
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
        
        naive = NaiveCP(alpha=alpha+epsilon, q_lo=predictors[f'qr_{target}_lo'], q_hi=predictors[f'qr_{target}_hi'])
        naive.fit(X_int, Y_int, target)
        C_Naive[f'C_{target}'] = naive.predict_set(X_te)
        
        dummy_L = np.zeros((len(X_te), 8))
        dummy_U = np.zeros((len(X_te), 8))
        C_NCCKE[f'C_{target}'] = (dummy_L, dummy_U)
        C_CKE[f'C_{target}'] = (dummy_L, dummy_U)
        C_WSCP_Inexact[f'C_{target}'] = (dummy_L, dummy_U)
        C_WSCP_Exact[f'C_{target}'] = (dummy_L, dummy_U)

    Y_0_te = dataset_te['Y0_RR'].astype('float')
    Y_1_te = dataset_te['Y1_PFCA'].astype('float')
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
    n_te=1000,
    alpha=0.2,
    epsilon=0.05,
):
    
    print('data per fold - n_test: ' + str(n_te) + '  n_int: ' + str(n_int) + '  n_obs: ' + str(n_obs))
    fold_logs = []
    for fold_id in range(n_folds):
        print( 'Fold :'+str(fold_id))
        obs_idx=np.random.choice(len(dataset_obs['X']), size=n_obs, replace=False)
        idx_0 = np.where(dataset_int['T'] == 0)[0]
        idx_1 = np.where(dataset_int['T'] == 1)[0]
        c0 = np.random.choice(idx_0, size=min(len(idx_0), n_int // 2), replace=False)
        c1 = np.random.choice(idx_1, size=min(len(idx_1), n_int - len(c0)), replace=False)
        int_idx = np.concatenate([c0, c1])
        te_idx=np.random.choice(len(dataset_te['X']), size=n_te, replace=False)
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
    }

def _fmt_param(value):
    return str(value).replace(".", "p")

def results_path(results_dir="../results", alpha=0.2, epsilon=0.05, confounding_strength=None):
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

def save_results(results, results_dir="../results", confounding_strength=None):
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
    residual = Y.astype(float)- X[:, :8].astype(float)
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

if __name__ == '__main__':
    
    for confounding_strength in [0,0.25,0.5,0.75,1]:
        print('Running Experiments with Confounding Strength: ' + str(confounding_strength))
        dataset = load_and_apply_treatment_policy(
            data_path="../data/scheduling_data.npy",
            T=1.,
            confounding_strength=confounding_strength,
            interventional_frac=0.1,
            seed=1,
        )
        mask_int = dataset["is_interventional"] == 1
        mask_obs = dataset["is_interventional"] == 0
        dataset_int = {}
        dataset_obs = {}
        for key, value in dataset.items():
            value = np.asarray(value)
            dataset_int[key] = value[mask_int]
            dataset_obs[key] = value[mask_obs]
        
        n_tr = 5000
        n_te = 5000
        dataset_train = {key: value[:n_tr] for key, value in dataset_obs.items()}
        dataset_te = {key: value[n_tr:n_tr+n_te] for key, value in dataset_obs.items()}
        dataset_cal = {key: value[n_tr+n_te:] for key, value in dataset_obs.items()}
        alpha = 0.1
        epsilon = 0.025
        alpha_base_pred=alpha
        is_RR = dataset_train['T'] == 0
        print('Total number of RR data : '+str(np.sum(is_RR))+'  --  Total number of PFCA data : '+str(np.sum(~is_RR)))
        qr_0_lo,qr_0_hi = fit_base_predictors( dataset_train['X'][is_RR], dataset_train['Y_obs'][is_RR], alpha_base_pred)
        qr_1_lo,qr_1_hi = fit_base_predictors(dataset_train['X'][~is_RR], dataset_train['Y_obs'][~is_RR], alpha_base_pred)
        
        pr_score_est=fit_likelihood_ratio_estimator(dataset_train['X'],dataset_train['T'])
        
        n_tr_it=int(0*n_tr)
        if n_tr_it>0:
            dataset_train_it = {key: value[:n_tr_it] for key, value in dataset_int.items()}
            dataset_cal_int = {key: value[n_tr_it:] for key, value in dataset_int.items()}
            is_RR_it = dataset_train_it['T'] == 0
            qr_it_0_lo, qr_it_0_hi = fit_base_predictors( dataset_train_it['X'][is_RR_it], dataset_train_it['Y_obs'][is_RR_it], alpha_base_pred)
            qr_it_1_lo, qr_it_1_hi = fit_base_predictors(dataset_train_it['X'][~is_RR_it], dataset_train_it['Y_obs'][~is_RR_it], alpha_base_pred)
            predictors={'qr_0_lo':qr_0_lo,'qr_0_hi':qr_0_hi,'qr_1_lo': qr_1_lo,'qr_1_hi': qr_1_hi,'qr_it_0_lo':qr_1_lo,'qr_it_0_hi':qr_it_0_hi,'qr_it_1_lo': qr_it_1_lo,'qr_it_1_hi': qr_it_1_hi,'pr_score_est': pr_score_est}
        else:
            dataset_cal_int = dataset_int
            predictors={'qr_0_lo':qr_0_lo,'qr_0_hi':qr_0_hi,'qr_1_lo': qr_1_lo,'qr_1_hi': qr_1_hi,'pr_score_est': pr_score_est}
        seed=0
        results = run_experiment(
            dataset_cal,
            dataset_cal_int,
            dataset_te,
            predictors,
            dataset_obs_train=dataset_train,
            n_folds=100,
            alpha=alpha,
            epsilon=epsilon,
        )
        saved_results_path = save_results(
            results,
            confounding_strength=confounding_strength,
        )
        print("Saved results to: " + saved_results_path)
        plot_results(results, alpha=alpha, epsilon=epsilon, confounding_strength=confounding_strength)
