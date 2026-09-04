
class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, state, reward, done):
        return self.action_space.sample()

    def seed(self, seed=0):
        self.action_space.seed(seed)
