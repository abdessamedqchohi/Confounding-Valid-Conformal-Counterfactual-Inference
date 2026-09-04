import random
from math import floor, ceil

import numpy as np
from gym import spaces, Env
from scipy import constants

from ..utils.misc import calculate_thermal_noise
from ..utils.prop_model import PropModel

class TimeFreqResourceAllocationV0(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    bw_mhz = 5  
    max_pkt_size_bits = 5096
    x_max_m = 1000
    y_max_m = 1000

    SINR_COEFF = 8  

    def __init__(self, n_ues=32, n_prbs=25, buffer_max_size=32, eirp_dbm=13, f_carrier_mhz=2655,
                 max_pkt_size_bits=41250, it=10, t_max=65536):
        super().__init__()
        self._seed = None
        self.K = n_ues  
        self.Nf = n_prbs  
        self.L = buffer_max_size  
        self.it = it  
        self.EIRP_DBM = eirp_dbm
        self.f_carrier_mhz = f_carrier_mhz  
        self.max_pkt_size_bits = max_pkt_size_bits
        self.t_max = t_max
        self.tti_max = ceil(t_max/n_prbs)

        self.bts_pos = [self.x_max_m / 2, self.y_max_m / 2]
        self.propagation_model = PropModel(self.f_carrier_mhz)
        self.n_mw = calculate_thermal_noise(self.bw_mhz * 1E-6)

        self.low = np.array([0] * self.K +  
                            [0] * self.K * self.L +  
                            [0] * self.K * self.L +  
                            [0, 0, 0, 0] * self.K +  
                            [0])  
        self.high = np.array([15] * self.K +  
                             [self.max_pkt_size_bits] * self.K * self.L +  
                             [self.tti_max] * self.K * self.L +  
                             [1, 1, 1, 1] * self.K +  
                             [self.Nf - 1])  
        self.observation_space = spaces.Box(self.low, self.high, dtype=np.uint32)

        self.action_space = spaces.Discrete(self.K)
        self.reward_range = (0, 1)

        self.cqi = None
        self.s = None  
        self.e = None  
        self.qi = None
        self.p = 0

        self.t = 0  
        self.tti = 0  
        self.ue_pos = None  
        self.ue_v_mps = None  
        self.ue_dir = None  
        self.spectral_efficiency = None
        self.tti_next_pkt = None  

        self.reset()

        assert self.K % 4 == 0, "K must be a multiple of 4 in order to have the same number of UEs per QoS class."

    def reset(self):
        self.cqi = np.zeros(shape=(self.K,), dtype=np.uint8)
        self.s = np.zeros(shape=(self.K, self.L), dtype=np.uint32)
        self.e = np.zeros(shape=(self.K, self.L), dtype=np.uint32)

        possible_qi = np.array([
            [0, 0, 0, 1],  
            [0, 0, 1, 0],  
            [0, 1, 0, 0],  
            [1, 0, 0, 0]  
        ])

        random_indices = np.random.choice(4, size=self.K)
        
        self.qi = possible_qi[random_indices]
        
        self.qi = np.repeat(np.array([[1, 0, 0, 0]]), self.K, axis=0)
        self.p = 0
        self.t = 0
        self.tti = 0
        self.ue_pos = np.random.uniform([0, 0], [self.x_max_m, self.y_max_m], size=(self.K, 2))  
        self.ue_v_mps = np.random.normal(1.36, scale=0.19, size=(self.K,))  
        self.ue_dir = np.random.uniform(0, 2 * constants.pi, size=(self.K,))  
        self.spectral_efficiency = np.zeros(shape=(self.K,))
        self.tti_next_pkt = np.random.randint(8, size=(self.K,))*0  
        self._recalculate_rf()
        self._generate_traffic()
        self._update_state()
        mask = self.s > 0
        n = mask.sum()
        
        self.e[mask] += np.random.randint(0, 50,size=n,dtype=self.e.dtype)
        return np.array(self.state)

    def seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        self.propagation_model.seed(seed=seed)
        self._seed = seed

    def step(self, action):
        assert self.action_space.contains(action), f"{action} ({type(action)}) invalid"

        if np.sum(self.s[action, :]) > 0:  
            
            mask = (self.s[action, :] > 0)
            subset_idx = np.argmax(self.e[action, mask])
            l_old = np.arange(self.L)[mask][subset_idx]

            assert self.s[action, l_old] > 0, f"t={self.t}. Oldest packet has size {self.s[action, l_old]} " +                                              f"and age {self.e[action, l_old]}. " +                                              f"User has {np.sum(self.s[action, :])} bits in buffer."  
            tx_data_bits = floor(
                self.spectral_efficiency[action] * self.bw_mhz / self.Nf * 1E3)  
            while tx_data_bits > 0 and self.s[action, l_old] > 0:  
                if tx_data_bits >= self.s[action, l_old]:  
                    tx_data_bits -= self.s[action, l_old]
                    self.s[action, l_old] = 0
                    self.e[action, l_old] = 0
                    l_old = np.argmax(self.e[action, :])  
                else:  
                    self.s[action, l_old] -= tx_data_bits
                    break

        reward = 0
        self.t += 1  
        self.p = self.t % self.Nf  
        if self.p == 0:
            reward = self._calculate_reward()
            self.tti += 1  
            self.e[self.s > 0] += 1  
            
            self._move_ues()
            self._recalculate_rf()

        self._update_state()
        done = bool(self.t >= self.t_max)
        return np.array(self.state), reward, done, {}

    def render(self, mode='human', close=False):
        pass

    def _calculate_reward(self):
        r_gbr = 0
        r_non_gbr = 0

        for u, qi in enumerate(self.qi):
            gbr_delayed_pkts = np.array([])
            non_gbr_pkts = np.array([])
            non_gbr_delayed_pkts = np.array([])
            if np.array_equal(qi, [0, 0, 0, 1]):
                gbr_delayed_pkts = np.where(self.e[u, :] > 100)[0]
            elif np.array_equal(qi, [0, 0, 1, 0]):
                gbr_delayed_pkts = np.where(self.e[u, :] > 150)[0]
            elif np.array_equal(qi, [0, 1, 0, 0]):
                gbr_delayed_pkts = np.where(self.e[u, :] > 30)[0]
            elif np.array_equal(qi, [1, 0, 0, 0]):
                non_gbr_delayed_pkts = np.where(self.e[u, :] > 300)[0]
                non_gbr_pkts = np.where(self.s[u, :] > 0)[0]

            if gbr_delayed_pkts.size > 0:
                r_gbr += np.sum(self.s[u, gbr_delayed_pkts])

            if non_gbr_delayed_pkts.size > 0:
                r_non_gbr += np.sum(self.s[u, non_gbr_delayed_pkts])
            if non_gbr_pkts.size > 0:
                r_non_gbr += np.sum(self.s[u, non_gbr_pkts])

        return -r_gbr - r_non_gbr

    def _move_ues(self):
        d_m = self.ue_v_mps * 1E-3  
        delta_x = d_m * np.cos(self.ue_dir)
        delta_y = d_m * np.sin(self.ue_dir)

        for u, pos in enumerate(self.ue_pos):
            if pos[0] + delta_x[u] > self.x_max_m or pos[0] + delta_x[u] < 0:
                delta_x[u] = -delta_x[u]
                self.ue_dir[u] = np.random.uniform(0, 2 * constants.pi)  
            if pos[1] + delta_y[u] > self.y_max_m or pos[1] + delta_y[u] < 0:
                delta_y[u] = -delta_y[u]
                self.ue_dir[u] = np.random.uniform(0, 2 * constants.pi)  

        self.ue_pos[:, 0] += delta_x
        self.ue_pos[:, 1] += delta_y

    def _recalculate_rf(self):
        distances_m = np.linalg.norm(self.ue_pos - self.bts_pos, axis=1)
        pathloss_db = self.propagation_model.get_free_space_pl_db(distances_m, shadowing_db=6)
        rx_pwr_dbm = self.EIRP_DBM - pathloss_db  
        self._calculate_spectral_efficiency(rx_pwr_dbm)
        self._spectral_efficiency_to_cqi()

    def _calculate_spectral_efficiency(self, rx_pwr_dbm):
        interference_dbm = -105  

        p_mw = (10 ** (rx_pwr_dbm / 10))  
        interference_mw = 10 ** (interference_dbm / 10)

        sinr = p_mw / (self.n_mw + interference_mw)
        se = np.log2(1 + sinr / self.SINR_COEFF)  

        self.spectral_efficiency = np.clip(se, 0, 9.6)  

    def _spectral_efficiency_to_cqi(self):
        
        self.cqi[np.where(self.spectral_efficiency <= 0.1523)] = 0
        self.cqi[np.where((0.1523 < self.spectral_efficiency) & (self.spectral_efficiency <= 0.2344))] = 1
        self.cqi[np.where((0.2344 < self.spectral_efficiency) & (self.spectral_efficiency <= 0.3770))] = 2
        self.cqi[np.where((0.3770 < self.spectral_efficiency) & (self.spectral_efficiency <= 0.6016))] = 3
        self.cqi[np.where((0.6016 < self.spectral_efficiency) & (self.spectral_efficiency <= 0.8770))] = 4
        self.cqi[np.where((0.8770 < self.spectral_efficiency) & (self.spectral_efficiency <= 1.1758))] = 5
        self.cqi[np.where((1.1758 < self.spectral_efficiency) & (self.spectral_efficiency <= 1.4766))] = 6
        self.cqi[np.where((1.4766 < self.spectral_efficiency) & (self.spectral_efficiency <= 1.9141))] = 7
        self.cqi[np.where((1.9141 < self.spectral_efficiency) & (self.spectral_efficiency <= 2.4063))] = 8
        self.cqi[np.where((2.4063 < self.spectral_efficiency) & (self.spectral_efficiency <= 2.7305))] = 9
        self.cqi[np.where((2.7305 < self.spectral_efficiency) & (self.spectral_efficiency <= 3.3223))] = 10
        self.cqi[np.where((3.3223 < self.spectral_efficiency) & (self.spectral_efficiency <= 3.9023))] = 11
        self.cqi[np.where((3.9023 < self.spectral_efficiency) & (self.spectral_efficiency <= 4.5234))] = 12
        self.cqi[np.where((4.5234 < self.spectral_efficiency) & (self.spectral_efficiency <= 5.1152))] = 13
        self.cqi[np.where((5.1152 < self.spectral_efficiency) & (self.spectral_efficiency <= 5.5547))] = 14
        self.cqi[np.where(5.5547 < self.spectral_efficiency)] = 15

    def _generate_traffic(self):
        for u, qi in enumerate(self.qi):
            if self.tti == self.tti_next_pkt[u]:
                buffer_gaps = np.where(self.s[u, :] == 0)[0]  
                if buffer_gaps.size == 0:  
                    print(f"Buffer overflow. Disregarding new GBR (Conversational Voice) packet for UE {u}.")
                    g = None
                else:
                    g = buffer_gaps[0]  
                    self.e[u, g] = 0  

                if np.array_equal(qi, [0, 0, 0, 1]):  
                    if buffer_gaps.size > 0:
                        self.s[u, g] = 584
                    self.tti_next_pkt[u] = self.tti + 20
                elif np.array_equal(qi, [0, 0, 1, 0]):  
                    
                    if buffer_gaps.size > 0:
                        self.s[u, g] = 41250
                    self.tti_next_pkt[u] = self.tti + 33
                elif np.array_equal(qi, [0, 1, 0, 0]):  
                    if buffer_gaps.size > 0:
                        self.s[u, g] = 200
                    self.tti_next_pkt[u] = self.tti + 20
                elif np.array_equal(qi, [1, 0, 0, 0]):  
                    
                    if buffer_gaps.size > 0:
                        self.s[u, g] = min(max(1, np.random.geometric(1 / 20000)), self.max_pkt_size_bits)
                    self.tti_next_pkt[u] = self.tti + np.random.geometric(1 / self.it)

                if buffer_gaps.size > 0:
                    assert 1 <= self.s[u, g] <= self.max_pkt_size_bits, f"Packet size {self.s[u, g]} out of range."

    def _update_state(self):
        self.state = np.concatenate((self.cqi, self.s.flatten(), self.e.flatten(), self.qi.flatten(), [self.p]))