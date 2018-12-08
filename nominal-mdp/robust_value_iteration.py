### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

from scipy.io import savemat
from test_env.envs.common import render_state
import matplotlib.pyplot as plt
import numpy as np
import gym
import time
from test_env import *
from likelihood import SigmaLikelihood
from entropy import SigmaEntropy
from value_iteration import value_iteration
import os
import argparse

np.set_printoptions(precision=3)

EPSILON = 0.1
np.set_printoptions(suppress=True)
np.set_printoptions(linewidth=200)

parser = argparse.ArgumentParser()
parser.add_argument('--grid_size', type=int)
parser.add_argument('--k', type=int)
parser.add_argument('--run_id', type=int)
parser.add_argument('--g2g', type=float)
parser.add_argument('--b2b', type=float)
parser.add_argument('--epsilon', type=float)
parser.add_argument('--cost_storm', type=float)

parser.add_argument('--action', type=str)  # plot_robust / plot_nominal / save_data
config = parser.parse_args()

'''
config = Hehe()
config.grid_size = 11
config.k = 1
config.run_id = 0
config.g2g = 0.9
config.b2b = 0.1
config.epsilon = 0.75
'''


def RobustBellmanOp(P, Sigma, state, action, gamma):
    """Represent R(s,a) + gamma * Sigma
    Notice that R(s,a) is the expected cost of execute |a| at state s.
    Returns float value

    Ve is the estimated robust value function.
    Returns
    -------
    value function of state: float
    	The value function of state.
    """
    BV = 0
    # Here there is a strong assumption that
    #   sum(p(s'|(s,a)) * R(s,a,s')) = R(s,a,s') = R(s,a)
    for t in P[state][action]:
        probability = t[0]
        nextstate = t[1]
        cost = t[2]
        done = t[3]
        # BV += probability * cost

        if done:
            BV += probability * cost
        else:
            BV += probability * (cost + gamma * Sigma[state, action])
    return BV


def robust_value_iteration(robust_algorithm, P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3, tol2=1.0):
    """
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P: dictionary
		It is from gym.core.Environment
		P[state][action] is tuples with (probability, nextstate, cost, terminal)
	nS: int
		number of states
	nA: int
		number of actions
	gamma: float
		Discount factor. Number in range [0, 1)
	max_iteration: int
		The maximum number of iterations to run before stopping. Feel free to change it.
	tol: float
		Determines when value function has converged.
	Returns:
	----------
	value function: np.ndarray
	policy: np.ndarray
	"""
    V = np.zeros(nS)
    V.fill(1000.)
    sigma = np.zeros((nS, nA))
    policy = np.zeros(nS, dtype=int)
    for iter_count in range(max_iteration):
        # print(iter_count)

        # Need to estimate sigma, which is of dimension |nS|*|nA|
        # This can simply be p^T V for now.
        # SigmaLikelihood(P, V, nS, nA, sigma, tol2, iter_count)
        # SigmaEntropy(P, V, nS, nA, sigma, tol2)
        robust_algorithm(P, V, nS, nA, sigma, tol2)

        newV = np.zeros(nS)
        for state in range(nS):
            BV = np.zeros(nA)
            for action in range(nA):
                BV[action] = RobustBellmanOp(P, sigma, state, action, gamma)
            newV[state] = BV.min()
        # Calculate difference of the value functions.
        Vdiff = np.max(np.abs(newV - V))
        V = newV
        if Vdiff < tol:
            break
    # Calculate the policy.
    for state in range(nS):
        BV = np.zeros(nA)
        for action in range(nA):
            BV[action] = RobustBellmanOp(P, sigma, state, action, gamma)
        policy[state] = np.argmin(BV)
    return V, policy, sigma

def render_single(env, policy, seed_feed=99, if_render=True, iter_tot=100):
    """Renders policy once on environment. Watch your agent play!

		Parameters
		----------
		env: gym.core.Environment
			Environment to play on. Must have nS, nA, and P as
			attributes.
		Policy: np.array of shape [env.nS]
			The action to take at a given state
	"""

    episode_reward = 0
    ob_list = []
    ob = env.reset()
    ob_list.append(ob)
    env.seed(seed_feed)
    for t in range(iter_tot):
        if if_render:
            env.render()
            time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        ob_list.append(ob)
        episode_reward += rew
        if done:
            break
    # assert done
    if if_render:
        env.render()
        print("Episode cost: %f" % episode_reward)
    return episode_reward, ob_list

def main_experiments():
    '''
    env = gym.make("AirCraftRouting-v4")
    env.set(config)
    render_state(env.nS - 1, env.nrow, env.ncol, env.storm_maps,
            env.terminal_pos)
    V_vi, p_vi, sigma_vi = robust_value_iteration(
        env.Q, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3)
    V_old, p_old = value_iteration(
        env.Q, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3)
    print("-------------- Value of robust --------------")
    print(V_vi.reshape((2, 5, 5)))
    print(p_vi.reshape((2, 5, 5)))
    print("-------------- Value of nomial --------------")
    print(V_old.reshape((2, 5, 5)))
    print(p_old.reshape((2, 5, 5)))
    import ipdb
    ipdb.set_trace()
    '''

    if config.action.startswith('plot_'):
        env = gym.make("AirCraftRouting-v4")
        env.set(config)
        ret_robust = []
        ret_normial = []
        V_vi, p_vi, sigma_vi = robust_value_iteration(
            SigmaLikelihood, env.Q, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3, tol2=config.epsilon / 0.45)
        V_old, p_old = value_iteration(
            env.Q, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3)
        exp_tot = 3
        if env.tk == 2 and config.action == 'plot_robust':
            # Without the storm
            cloth = np.concatenate((
                V_vi.reshape((env.tk, config.grid_size, config.grid_size))[0],
                np.ones((config.grid_size, 1)) * V_vi.reshape((env.tk, config.grid_size, config.grid_size))[0].max(),
                V_old.reshape((env.tk, config.grid_size, config.grid_size))[0],
                ), 1)
            plt.imshow(cloth)
            plt.show()

            # With the storm
            cloth = np.concatenate((
                V_vi.reshape((env.tk, config.grid_size, config.grid_size))[1],
                np.ones((config.grid_size, 1)) * V_vi.reshape((env.tk, config.grid_size, config.grid_size))[1].max(),
                V_old.reshape((env.tk, config.grid_size, config.grid_size))[1],
                ), 1)
            plt.imshow(cloth)
            plt.show()
        '''
        savemat('values.mat', {'v_nominal': V_old.reshape((env.tk, config.grid_size, config.grid_size)), 'v_robust': V_vi.reshape((env.tk, config.grid_size, config.grid_size))})
        exit()
        import ipdb
        ipdb.set_trace()
        '''
        for j in range(exp_tot):
            if config.action == 'plot_robust':
                print('robust')
                ret, _ = render_single(env, p_vi, j, True, iter_tot=1000)
                ret_robust.append(ret)
            if config.action == 'plot_nominal':
                print('nominal')
                ret, _ = render_single(env, p_old, j, True, iter_tot=1000)
                ret_normial.append(ret)
            print("Finish exp iter %d out of %d" % (j+1, exp_tot))
            import ipdb
            ipdb.set_trace()
        if config.action == 'plot_robust':
            print(sum(ret_robust) / float(exp_tot))
        if config.action == 'plot_nominal':
            print(sum(ret_normial) / float(exp_tot))
    elif config.action == 'sweep':
        '''  Setting 1 get 20x20
        epsilon_tot = 20
        cost_tot = 20
        exp_tot = 30

        epsilon_max = min(config.g2g, 1. - config.b2b)
        epsilon_sweep = np.arange(0.001, epsilon_max, epsilon_max / float(epsilon_tot + 2))
        cost_sweep = np.arange(1., float(1. + epsilon_tot), 1.)
        '''

        ''' Setting 2 get 200x1
        epsilon_tot = 200
        cost_tot = 1
        exp_tot = 30

        epsilon_max = min(config.g2g, 1. - config.b2b)
        epsilon_sweep = np.arange(0.001, epsilon_max, epsilon_max / float(epsilon_tot + 2))
        cost_sweep = np.array([5.])
        '''

        # Setting 3 get 1x200
        epsilon_tot = 1
        cost_tot = 200
        exp_tot = 30

        epsilon_sweep = np.array([0.328])
        cost_sweep = np.arange(1., 14., 13./ float(cost_tot + 2))

        ret_robust = np.zeros((epsilon_tot, cost_tot, exp_tot), dtype=float)
        ret_nominal = np.zeros((epsilon_tot, cost_tot, exp_tot), dtype=float)
        epsilon_record = np.zeros((epsilon_tot, ), dtype=float)
        cost_record = np.zeros((cost_tot, ), dtype=float)

        for epsilon_index in range(epsilon_tot):
            for cost_index in range(cost_tot):
                config.epsilon = epsilon_sweep[epsilon_index]
                config.cost_storm = cost_sweep[cost_index]
                env = gym.make('AirCraftRouting-v4')
                env.set(config)
                print('epsilon_index %d out of %d, epsilon=%.4f. cost_index %d out of %d, cost=%.4f.' \
                        % (epsilon_index, epsilon_tot, config.epsilon, cost_index, cost_tot, config.cost_storm))

                _, p_old = value_iteration(
                        env.Q, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3)
                _, p_robust, _ = robust_value_iteration(
                        SigmaLikelihood, env.Q, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3, tol2=epsilon_sweep[epsilon_index] / 0.45)

                epsilon_record[epsilon_index] = config.epsilon
                cost_record[cost_index] = config.cost_storm

                for exp_index in range(exp_tot):
                    print('  exp_index %d out of %d' % (exp_index, exp_tot))
                    ret_robust[epsilon_index, cost_index, exp_index], _ = render_single(env, p_robust, exp_index, False, iter_tot=1000)
                    ret_nominal[epsilon_index, cost_index, exp_index], _ = render_single(env, p_old, exp_index, False, iter_tot=1000)

        savemat('GS_%d_K_%d_RUN_%d_G2G_%.2f_B2B_%.2f_sweep_%dby%d.mat' % (config.grid_size, config.k, config.run_id, config.g2g, config.b2b, epsilon_tot, cost_tot), {'epsilon_record': epsilon_record, 'cost_record': cost_record, 'ret_robust': ret_robust, 'ret_nominal': ret_nominal})
    else:
        print('LOL')


if __name__ == '__main__':
    main_experiments()
