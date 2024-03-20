"""Environment for training the acceleration behavior of vehicles in a ring."""

from flow.core import rewards
from flow.envs.base import Env

from gym.spaces.box import Box

import numpy as np

ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    'max_accel': 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    'max_decel': 3,
    # desired velocity for all vehicles in the network, in m/s
    'target_velocity': 10
}


class CustomAccelEnv(Env):
    """Fully observed acceleration environment.

    Required from env_params:

    * max_accel: maximum acceleration for autonomous vehicles, in m/s^2
    * max_decel: maximum deceleration for autonomous vehicles, in m/s^2
    * target_velocity: desired velocity for all vehicles in the network, in m/s

    States
        The state consists of the velocities and absolute position of all
        vehicles in the network. This assumes a constant number of vehicles.

    Actions
        Actions are a list of acceleration for each rl vehicles, bounded by the
        maximum accelerations and decelerations specified in EnvParams.

    """

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        for p in ADDITIONAL_ENV_PARAMS.keys():
            if p not in env_params.additional_params:
                raise KeyError(
                    'Environment parameter \'{}\' not supplied'.format(p))


        super().__init__(env_params, sim_params, network, simulator)

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1, ),
            dtype=np.float32)

    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity']
        return Box(
            low=-1000000, # an arbitrary high enough number
            high=1000000, # an arbitrary high enough number
            shape=(1, ),
            dtype=np.float32)




