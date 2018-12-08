### MDP Value Iteration and Policy Iteratoin
# You might not need to use all parameters

import numpy as np
import gym
import time
from test_env import *
import os

np.set_printoptions(precision=3)


def BellmanOp(P, V, state, action, gamma):
    """Represent R(s,a) + gamma * sum(p(s'|(s,a)) * V(s'))
    Notice that R(s,a) is the expected cost of execute |a| at state s.
    Returns float value

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
        if done:
            BV += probability * (cost)
        else:
            # TODO(yejiayu): Here we need to work on the robust part.
            BV += probability * (cost + gamma * V[nextstate])
    return BV


def value_iteration(P, nS, nA, gamma=0.9, max_iteration=20, tol=1e-3):
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
    V = np.zeros((nS,), dtype=float)
    V.fill(1000)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    ############################
    for iter_count in range(max_iteration):
        newV = np.zeros(nS)
        for state in range(nS):
            BV = np.zeros(nA)
            for action in range(nA):
                BV[action] = BellmanOp(P, V, state, action, gamma)
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
            BV[action] = BellmanOp(P, V, state, action, gamma)
        policy[state] = np.argmin(BV)
    return V, policy


def example(env):
    """
    Show an example of gym
	Parameters
	----------
	env: gym.core.Environment
		Environment to play on. Must have nS, nA, and P as
		attributes.
    """
    env.seed(0)
    from gym.spaces import prng
    prng.seed(10)  # for print the location
    # Generate the episode
    ob = env.reset()
    for t in range(20):
        env.render()
        a = env.action_space.sample()
        ob, rew, done, _ = env.step(a)
        if done:
            break
    assert done
    env.render()


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
        env.render()
        time.sleep(0.5)  # Seconds between frames. Modify as you wish.
        a = policy[ob]
        ob, rew, done, _ = env.step(a)
        ob_list.append(ob)
        episode_reward += rew
        if done:
            break
    assert done
    if if_render:
        env.render()
    print("Episode cost: %f" % episode_reward)
    return episode_reward, ob_list


# Feel free to run your own debug code in main!
# Play around with these hyperparameters.
if __name__ == "__main__":
    # TODO: make this an arg.
    env = gym.make("AirCraftRoutingSimple-v1")
    print(env.__doc__)
    # print("Here is an example of state, action, cost, and next state")
    # example(env)
    # print(env.P)
    V_vi, p_vi = value_iteration(
        env.P, env.nS, env.nA, gamma=1, max_iteration=100, tol=1e-3)
    print(env.P)
    render_single(env, p_vi)
    print('------------ All the storm map ------------')
    print(env.storm_maps.max(0))
    print('-------------------------------------------')
    print('------------ Normal Storm Transformation Q ------------')
    print(env.Qmatrix)
    print('-------------------------------------------------------')
    print('------------ Robust Perturbed Transformation P ------------')
    print(env.Pmatrix)
    print('-----------------------------------------------------------')
