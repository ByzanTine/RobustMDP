import os
import glob
import numpy as np
import random
from machine_replacement_env import MachineReplacementEnv, MachineReplacementDirichletEnv

discount_factor = 0.8
'''
# uncertainty in rewards
env = MachineReplacementEnv(state_num=50)
terminal_status = 0
actions = np.array([0, 0, 0, 1, 0, 0])
i = 0

while terminal_status == 0:
    cost, terminal_status = env.step(actions[i])
    print i, cost
    i += 1
'''

# uncertanity in transitions
env = MachineReplacementDirichletEnv(rng_seed=2)
terminal_steps = 10
#actions = np.zeros([50])
actions =  np.array([0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0,])
#actions = np.random.randint(2, size=terminal_steps)
print actions

i = 0
while i < terminal_steps:
    cost, state = env.step(actions[i])
    print actions[i], cost, state
    i += 1
