import os
import glob
import numpy as np
import random


class MachineReplacementEnv(object):
    def __init__(self,
                 state_num=50,
                 rng_seed=0):
        self.state_num = state_num
        self.current_state = 1
        self.rng = rng_seed
        self.terminal_status = 0
        np.random.seed(self.rng)

    def step(self, action):
        if action == 0:
            if self.current_state < self.state_num:
                self.current_state += 1
                cost = np.random.normal(0, 1e-4) #N(0, 1e-4)
            else:
                self.current_state = 1
                cost = np.random.normal(100, 800)
        if action == 1:
            self.current_state = 1
            if self.current_state < self.state_num:
                cost = np.random.normal(130, 1)
            else:
                cost = np.random.normal(130, 20)
            self.terminal_status = 1
        return cost, self.terminal_status

    def reset(self):
        self.current_state = 1


class MachineReplacementDirichletEnv(object):
    def __init__(self,
                 rng_seed=0):
        self.current_state = 1
        self.rng = rng_seed
        np.random.seed(self.rng)

    def step(self, action):
        if action == 1:
            # update state
            if self.current_state == 8:
                self.current_state = np.random.choice([-1, -2, 8], p=[0.6, 0.1, 0.3])
            elif self.current_state > 0:
                self.current_state = np.random.choice([-1, -2, self.current_state+1], \
                                        p=[0.6, 0.1, 0.3])
            elif self.current_state == -2:
                self.current_state == np.random.choice([-1, -2], p=[0.6, 0.4])

            # the cost
            if self.current_state == -2:
                cost = 10
            elif self.current_state == -1:
                cost = 2
            elif self.current_state == 8:
                cost = 20
            else:
                cost = 0

        if action == 0:
            if self.current_state == 8:
                cost = 20
            elif self.current_state > 0:
                self.current_state += np.random.choice(np.arange(2), p=[0.2, 0.8])
                cost = 0
            elif self.current_state == -1:
                self.current_state += np.random.choice([-1, 1], p=[0.2, 0.8])
                cost = 0
            else:
                cost = 0

        return cost, self.current_state

    def reset(self):
        self.current_state = 1
