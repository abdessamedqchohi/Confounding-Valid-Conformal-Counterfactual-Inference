import numpy as np
from scipy import constants

class PropModel:

    def __init__(self, f_mhz, n=2):
        self.f_mhz = f_mhz
        self.n = n  

    def get_free_space_pl_db(self, d_m, shadowing_db=0):
        noise = np.random.normal(scale=shadowing_db, size=d_m.size)
        return self.n * 10 * np.log10(4 * constants.pi * d_m * self.f_mhz * 1E6 / constants.c) + noise

    def seed(self, seed=0):
        np.random.seed(seed)
