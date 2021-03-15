from tf_agents.trajectories import time_step as ts

class SpeedPenalty(gym.Wrapper):
    def __init__(self, env, diff_factor= 0.1):
        super(SpeedPenalty, self).__init__(env)
        self._diff_factor = diff_factor
        self.reset = self.env.reset
        
    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps
        

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if action is not None:
            reward += (action[1] - action[2]) * self._diff_factor

        return observation, reward, done, info

from tf_agents.trajectories import time_step as ts
class NegRewardPenalty(gym.Wrapper):
    def __init__(self, env, penalty=0.,max_neg_steps= 50,from_step=100):
        super(NegRewardPenalty, self).__init__(env)
        self._penalty = penalty
        self._max_neg_steps = max_neg_steps
        self._from_step = from_step
        self._neg_step_counter = None
    
    @property
    def _elapsed_steps(self):
        return self.env._elapsed_steps
        
    def reset(self):
        self._neg_step_counter = 0
        return self.env.reset()   

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if self._elapsed_steps  > self._from_step and reward < 0:
            self._neg_step_counter += 1
            if not done and self._neg_step_counter > self._max_neg_steps:
                done = True
                reward += self._penalty
        else:
            self._neg_step_counter = 0

        return observation, reward, done, info