from gym import utils
import numpy as np


bboxes = np.load('./GS_11_K_1_RUN_0.npy')
nrow = ncol = 11

air_map = np.zeros((nrow, ncol), dtype='U10')
for i in range(nrow):
    for j in range(ncol):
        air_map[i, j] = '.'

# E
air_map[(nrow - 1) / 2, -1] = 'E'

# S
for i in range(bboxes.shape[0]):
    left = int(bboxes[i, 0])
    right = int(bboxes[i, 1])
    top = int(bboxes[i, 2])
    bottom = int(bboxes[i, 3])
    air_map[top:bottom, left:right] = 'S'

# A
air_map[(nrow - 1) / 2, 0] = utils.colorize('A', 'red', highlight=True)

for i in range(nrow):
    print(air_map[i].tostring().decode('utf-8'))
