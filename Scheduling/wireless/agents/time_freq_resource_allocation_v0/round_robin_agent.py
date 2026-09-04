from wireless.agents.random_agent import RandomAgent
import numpy as np

class RoundRobinAgent(RandomAgent):
    def __init__(self, action_space, n_ues, buffer_max_size):
        RandomAgent.__init__(self, action_space)
        self.t = 0      

        self.K = n_ues              
        self.L = buffer_max_size    

    def act(self, state, reward, done):
        action = self.t % self.K
        self.t += 1
        return action
