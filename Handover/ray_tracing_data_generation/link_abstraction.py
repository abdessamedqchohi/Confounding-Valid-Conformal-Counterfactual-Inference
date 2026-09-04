from dataclasses import dataclass

import numpy as np

@dataclass
class LinkConfig:
    num_prbs: int = 50
    subcarriers_per_prb: int = 12
    subcarrier_spacing_hz: float = 30e3
    tx_power_dbm_per_bs: float = 43.0

    @property
    def num_subcarriers(self):
        return self.num_prbs * self.subcarriers_per_prb

def db_to_linear(x_db):
    return 10.0 ** (np.asarray(x_db) / 10.0)

def linear_to_db(x):
    return 10.0 * np.log10(np.asarray(x) + 1e-30)

def dbm_to_w(dbm):
    return 1e-3 * db_to_linear(dbm)

def w_to_dbm(watts):
    return 10.0 * np.log10(np.asarray(watts) / 1e-3 + 1e-30)

def tx_power_per_prb_w(config):
    return dbm_to_w(config.tx_power_dbm_per_bs) / config.num_prbs

def channel_gain_per_prb(physical, config):
    h_freq = np.asarray(physical["channel_frequency_response"])
    power_sc = np.abs(h_freq) ** 2
    num_subcarriers = power_sc.shape[-1]

    if num_subcarriers != config.num_subcarriers:
        raise ValueError(
            "CFR subcarrier count does not match LinkConfig: "
            f"{num_subcarriers} vs {config.num_subcarriers}"
        )

    return power_sc.reshape(
        *power_sc.shape[:-1],
        config.num_prbs,
        config.subcarriers_per_prb,
    ).mean(axis=-1)

def compute_rsrp(physical, config):
    gain_prb = channel_gain_per_prb(physical, config)
    rsrp_w = tx_power_per_prb_w(config) * np.mean(gain_prb, axis=-1)
    return {
        "rsrp_w": rsrp_w,
        "rsrp_dbm": w_to_dbm(rsrp_w),
    }
