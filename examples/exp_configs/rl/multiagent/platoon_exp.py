from copy import deepcopy
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
from flow.envs.multiagent import BilateralPlatoonEnv, UnilateralPlatoonEnv, FlatbedEnv, PloegEnv
from flow.networks import PlatoonHighwayNetwork
from flow.networks import HighwayNetwork
from flow.utils.registry import make_create_env
from ray.tune.registry import register_env


# configurable parameters

# seed for the environmental behaviour
seed = 1
# type of control model environment: "UnilateralPlatoonEnv" or "BilateralPlatoonEnv"
env_type = UnilateralPlatoonEnv
# size of state's time frame
state_time_frame = 1
# use a modified variant of the original reward function for training
modified_reward_function = False

####################################################################

# time horizon of a single rollout
HORIZON = 600

vehicles = VehicleParams()

vehicles.add(
    veh_id='follower4',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=0,
    ),
    num_vehicles=1,
    initial_speed=12,
    color='grey')
vehicles.add(
    veh_id='follower3',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=0,
    ),
    num_vehicles=1,
    initial_speed=12,
    color='cyan')
vehicles.add(
    veh_id='follower2',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=0,
    ),
    num_vehicles=1,
    initial_speed=12,
    color='green')
vehicles.add(
    veh_id='follower1',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=0,
    ),
    num_vehicles=1,
    initial_speed=12,
    color='yellow')
vehicles.add(
    veh_id='follower0',
    acceleration_controller=(RLController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=0,
    ),
    num_vehicles=1,
    initial_speed=12,
    color='white')
vehicles.add(
    veh_id='leader',
    acceleration_controller=(LeaderController, {}),
    routing_controller=(ContinuousRouter, {}),
    car_following_params=SumoCarFollowingParams(
        speed_mode='aggressive',
        min_gap=0,
    ),
    num_vehicles=1,
    initial_speed=12,
    color='red')


additional_net_params = ADDITIONAL_NET_PARAMS.copy()
additional_net_params.update({
    "length": 10000,
    "num_vehicles":6,
    "upper_gap_bound":50,
    "lower_gap_bound":30,
    "speed_limit": 36,
    "default_gaps": [30, 30, 30, 30, 30]

})


flow_params = dict(
    # name of the experiment
    exp_tag='multi_lane_highway',

    # name of the flow environment the experiment is running on
    env_name=env_type,

    # name of the network class the experiment is running on
    network=PlatoonHighwayNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        sim_step=0.1,
        render=False,
        restart_instance=True,
        initial_speed_variance=4,
        seed=seed
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            'max_accel': 3,
            'max_decel': 3,
            'num_scenarios':6,
            'state_time_frame': state_time_frame,
            'modified_reward_function': modified_reward_function
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





