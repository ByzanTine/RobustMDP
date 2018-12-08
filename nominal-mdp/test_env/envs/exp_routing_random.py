"""
A classic Aircraft routing problem described in Nilm 2005.
"""

import logging
import sys
import random
import os

import numpy as np
import pickle
from gym import utils
from gym.envs.toy_text import discrete

from math import factorial
import matplotlib.pyplot as plt
import math
from .common import render_state

logger = logging.getLogger(__name__)

# Action
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

class ExpRoutingRandom(discrete.DiscreteEnv):
    """
    The aircraft must go through a stormy area.

    The space is a N x N grid.

    Denote A as the aircraft, E as the end point.

                    X
                    X
    A               X           E
                    X
                    X

    Here Xs are a stormy area that has some chance of storm.

    For simplicity, we place A at the middle of very left.
    E at the middle of very right.

    The storm follow a transition matrix of 2^k * 2^k.
    Each entry is the probability transit from one possible state to the other.

    Assume k obstacle.
    """
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, desc=None):
        pass

    def set(self, config):
        GRID_SIZE = config.grid_size
        K = config.k
        assert 2 * K + 1 <= GRID_SIZE
        RUN_ID = config.run_id  # Generate and store on a differently generated randomness under the same setting
        MAP_FILE = 'random_store/GS_%d_K_%d_RUN_%d.npy' % (GRID_SIZE, K, RUN_ID)
        AREA_UP_BOUND = math.ceil(GRID_SIZE * GRID_SIZE / float(K))

        # Q Single
        Q_SINGLE_G2G = config.g2g
        Q_SINGLE_G2B = 1. - Q_SINGLE_G2G
        Q_SINGLE_B2B = config.b2b
        Q_SINGLE_B2G = 1. - Q_SINGLE_B2B
        # P Perturbation
        E = config.epsilon  # always added to 2B and decreased to 2G
        P_SINGLE_G2G = Q_SINGLE_G2G - E
        P_SINGLE_G2B = 1. - P_SINGLE_G2G
        P_SINGLE_B2B = Q_SINGLE_B2B + E
        P_SINGLE_B2G = 1. - P_SINGLE_B2B

        # Cost
        COST_NORMAL = 1.
        COST_STORM = config.cost_storm

        if os.path.isfile(MAP_FILE):
            each_storm_map_bbox = np.load(MAP_FILE)
        else:
            each_storm_map_bbox = np.zeros((K, 4), dtype=int)  # storm areas can overlap
            for j in range(K):
                area = np.inf
                while area > AREA_UP_BOUND or area <= 0:
                    bbox_left = np.random.randint(0, GRID_SIZE - 1)
                    bbox_right = np.random.randint(0, GRID_SIZE - 1)
                    bbox_left, bbox_right = min(bbox_left, bbox_right), max(bbox_left, bbox_right)
                    bbox_top = np.random.randint(0, GRID_SIZE - 1)
                    bbox_bottom = np.random.randint(0, GRID_SIZE - 1)
                    bbox_top, bbox_bottom = min(bbox_top, bbox_bottom), max(bbox_top, bbox_right)
                    area = (bbox_bottom - bbox_top) * (bbox_right - bbox_left)
                each_storm_map_bbox[j] = np.array([bbox_left, bbox_right, bbox_top, bbox_bottom])
            np.save(MAP_FILE, each_storm_map_bbox)

        self.nrow, self.ncol = nrow, ncol = (GRID_SIZE, GRID_SIZE)

        k = K
        tk = 2**k
        nS = nrow * ncol * tk
        nA = 4

        # Get the binary: 12 -> 01100 -> [False, True, True, False, False], suppose K is 5
        fs = '{0:0' + str(K) + 'b}'

        def stormID2BoolList(id):
            return [int(x) > 0 for x in fs.format(id)]

        # calc distribution probability, treat p to be the probability of flip, 1-p fr hold
        def calcTransProb(count_G2G, count_G2B, count_B2G, count_B2B, prob_G2G, prob_G2B, prob_B2G, prob_B2B):
            return (prob_G2G ** count_G2G) * (prob_G2B ** count_G2B) * (prob_B2G ** count_B2G) * (prob_B2B ** count_B2B)

        # (row, col, storm_index) to stateID
        def to_s(row, col, storm_index):
            return nrow * ncol * storm_index + row * ncol + col

        # step
        def inc(row, col, a):
            if a == 0:  # left
                col = max(col - 1, 0)
            elif a == 1:  # down
                row = min(row + 1, nrow - 1)
            elif a == 2:  # right
                col = min(col + 1, ncol - 1)
            elif a == 3:  # up
                row = max(row - 1, 0)
            return (row, col)

        terminal_pos = (nrow // 2, ncol - 1)

        # obtain_storm_maps
        storm_maps = np.zeros((tk, nrow, ncol)).astype('float64')
        for i in range(tk):
            bool_list = stormID2BoolList(i)
            for j in range(len(bool_list)):
                if bool_list[j]:  # j-th storm is on
                    storm_maps[i,
                            each_storm_map_bbox[j][2]:each_storm_map_bbox[j][3],
                            each_storm_map_bbox[j][0]:each_storm_map_bbox[j][1]
                            ] = 1.

        # Conventions: Q is the nominal distribution and P is the perturbed one
        # Define Qs: only care about the transformation between storms
        # Define Ps: only care about the transformation between storms
        Qs = np.zeros((tk, tk), np.float32)
        Ps = np.zeros((tk, tk), np.float32)
        for i in range(tk):
            bool_list_i = stormID2BoolList(i)
            for j in range(tk):
                bool_list_j = stormID2BoolList(j)
                count_G2G = sum([not bool_list_i[z] and not bool_list_j[z] for z in range(k)])
                count_G2B = sum([not bool_list_i[z] and bool_list_j[z] for z in range(k)])
                count_B2G = sum([bool_list_i[z] and not bool_list_j[z] for z in range(k)])
                count_B2B = sum([bool_list_i[z] and bool_list_j[z] for z in range(k)])
                Qs[i, j] = calcTransProb(count_G2G, count_G2B, count_B2G, count_B2B,
                        Q_SINGLE_G2G, Q_SINGLE_G2B, Q_SINGLE_B2G, Q_SINGLE_B2B)
                Ps[i, j] = calcTransProb(count_G2G, count_G2B, count_B2G, count_B2B,
                        P_SINGLE_G2G, P_SINGLE_G2B, P_SINGLE_B2G, P_SINGLE_B2B)

        # Put Ps into P, Qs into Q:
        Q = {s: {a: [] for a in range(nA)} for s in range(nS)}
        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        Qmatrix = np.zeros((nS, nA, nS), dtype=np.float64)
        Pmatrix = np.zeros((nS, nA, nS), dtype=np.float64)
        for row in range(nrow):
            for col in range(ncol):
                for old_storm_id in range(tk):
                    old_state_id = to_s(row, col, old_storm_id)
                    # Note: cost is irrelavant to the new state!
                    if storm_maps[old_storm_id, row, col] == 1:
                        cost = COST_STORM
                    else:
                        cost = COST_NORMAL
                    for new_storm_id in range(tk):
                        for a in range(4):
                            newrow, newcol = inc(row, col, a)
                            new_state_id = to_s(newrow, newcol, new_storm_id)
                            done1 = (row, col) == terminal_pos
                            done2 = (newrow, newcol) == terminal_pos
                            P[old_state_id][a].append(
                                (Ps[old_storm_id, new_storm_id], new_state_id,
                                 cost, done1 or done2))
                            Pmatrix[old_state_id, a, new_state_id] = Ps[
                                old_storm_id][new_storm_id]
                            Q[old_state_id][a].append(
                                (Qs[old_storm_id, new_storm_id], new_state_id,
                                 cost, done1 or done2))
                            Qmatrix[old_state_id, a, new_state_id] = Qs[
                                old_storm_id][new_storm_id]

        # Take home
        self.storm_maps = storm_maps
        self.Qs = Qs
        self.Q = Q
        self.Qmatrix = Qmatrix
        self.Ps = Ps
        self.P = P
        self.Pmatrix = Pmatrix

        self.k = k
        self.nrow = nrow
        self.ncol = ncol
        self.terminal_pos = terminal_pos
        self.nS = nS
        self.nA = nA
        self.tk = tk

        # new object
        isd = np.zeros((tk, nrow, ncol)).astype('float64')
        isd[0, nrow // 2, 0] = 1
        super(ExpRoutingRandom, self).__init__(nS, nA, P, isd)

    def reset(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return super(ExpRoutingRandom, self).reset()

    def render(self, random='human', close=False):
        render_state(self.s, self.nrow, self.ncol, self.storm_maps, self.terminal_pos)
