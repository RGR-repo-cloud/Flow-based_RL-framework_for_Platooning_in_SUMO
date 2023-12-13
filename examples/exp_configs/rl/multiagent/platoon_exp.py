from copy import deepcopy
from ray.rllib.agents.ppo.ppo_policy import PPOTFPolicy
from flow.controllers import ContinuousRouter
from flow.controllers import IDMController
from flow.controllers import RLController
from flow.controllers import LeaderController
from flow.core.params import EnvParams
from flow.core.params import InitialConfig
from flow.core.params import NetParams
from flow.core.params import SumoParams
from flow.core.params import SumoCarFollowingParams
from flow.core.params import VehicleParams
from flow.networks.platoon_highway import ADDITIONAL_NET_PARAMS
from flow.envs.multiagent import PlatoonEnv
from flow.networks import PlatoonHighwayNetwork
from flow.networks import HighwayNetwork
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env

# time horizon of a single rollout
HORIZON = 3000
# number of rollouts per training iteration
N_ROLLOUTS = 2
# number of parallel workers
N_CPUS = 3

# We place one autonomous vehicle and 13 human-driven vehicles in the network
vehicles = VehicleParams()

vehicles.add(
    veh_id='follower4',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
    ),
    num_vehicles=1,
    initial_speed=0,
    color='grey')
vehicles.add(
    veh_id='follower3',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
    ),
    num_vehicles=1,
    initial_speed=0,
    color='cyan')
vehicles.add(
    veh_id='follower2',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
    ),
    num_vehicles=1,
    initial_speed=0,
    color='green')
vehicles.add(
    veh_id='follower1',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
    ),
    num_vehicles=1,
    initial_speed=0,
    color='yellow')
vehicles.add(
    veh_id='follower0',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
    ),
    num_vehicles=1,
    initial_speed=0,
    color='white')
vehicles.add(
    veh_id='leader',
    acceleration_controller=(LeaderController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
    ),
    num_vehicles=1,
    initial_speed=0,
    color='red')


additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    "length": 1000,
    "num_vehicles":6,
    "initial_gaps": [20, 40, 20, 15, 40],
    "speed_limit": 5

})


flow_params = dict(
    # name of the experiment
    exp_tag='multi_lane_highway',

    # name of the flow environment the experiment is running on
    env_name=PlatoonEnv,

    # name of the network class the experiment is running on
    network=PlatoonHighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=True,
        restart_instance=True
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            'target_velocity': 1,
            'max_accel': 3,
            'max_decel': 3,
            'sort_vehicles': True
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        additional_params=additional_net_params,
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization/
    # reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(spacing='custom'),
)


create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space



