from gym.envs.registration import register

import gym
from test_env.envs import *

register(
    id='AirCraftRouting-v1',
    entry_point='test_env.envs.air_craft_routing:AirCraftRouting',
)

register(
    id='AirCraftRouting-v2',
    entry_point='test_env.envs.air_craft_routing_random:AirCraftRoutingRandom',
)

register(
    id='AirCraftRouting-v3',
    entry_point='test_env.envs.complex_routing_random:ComplexRoutingRandom',
)

register(
    id='AirCraftRouting-v4',
    entry_point='test_env.envs.exp_routing_random:ExpRoutingRandom',
)

register(
    id='AirCraftRoutingSimple-v1',
    entry_point=
    'test_env.envs.robust_air_routing_testcase:AirCraftRoutingSimple',
)
