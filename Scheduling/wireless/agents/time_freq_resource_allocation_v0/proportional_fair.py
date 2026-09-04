from wireless.agents.random_agent import RandomAgent
import numpy as np

class ProportionalFairAgent(RandomAgent):
    def __init__(self, action_space, n_ues, buffer_max_size):
        RandomAgent.__init__(self, action_space)
        self.t = 0      

        self.K = n_ues              
        self.L = buffer_max_size    
        self.n = np.zeros(n_ues)    

    def _calculate_priorities(self, cqi, o, b, buffer_size_per_ue):
        priorities = (1 + o) / b * buffer_size_per_ue / (1 + self.n)
        return priorities

    @staticmethod
    def parse_state(state, num_ues, max_pkts):
        s = np.reshape(state[num_ues:num_ues * (1 + max_pkts)], (num_ues, max_pkts))  
        buffer_size_per_ue = np.sum(s, axis=1)

        e = np.reshape(state[num_ues * (1 + max_pkts):num_ues * (1 + 2 * max_pkts)], (num_ues, max_pkts))  
        o = np.max(e, axis=1)  

        cqi = state[0:num_ues]

        qi_ohe = np.reshape(state[num_ues + 2 * num_ues * max_pkts:5 * num_ues + 2 * num_ues * max_pkts], (num_ues, 4))
        qi = np.array([np.where(r == 1)[0][0] for r in qi_ohe])  

        b = np.zeros(qi.shape)
        b[qi == 3] = 100
        b[qi == 2] = 150
        b[qi == 1] = 30
        b[qi == 0] = 300

        return o, cqi, b, buffer_size_per_ue

    def act(self, state, reward, done):
        o, cqi, b, buffer_size_per_ue = self.parse_state(state, self.K, self.L)

        priorities = self._calculate_priorities(cqi, o, b, buffer_size_per_ue)

        action = np.argmax(priorities)
        self.n[action] += 1

        self.t += 1
        return action
