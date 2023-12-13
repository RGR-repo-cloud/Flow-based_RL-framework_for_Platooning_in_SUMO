"""Environment for training the acceleration behavior of vehicles in a ring."""
import numpy as np
from gym.spaces import Box

from flow.core import rewards
from flow.envs.multiagent.custom_accel import CustomAccelEnv
from flow.envs.multiagent.base import MultiEnv


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 30,
}


class ConstantSpeedEnv2V(CustomAccelEnv, MultiEnv):

    def _apply_rl_actions(self, rl_actions):

        v1_action = rl_actions['v1']
        v2_action = rl_actions['v2']

        
        ids = ['v1_0', 'v2_0']
        actions = [v1_action, v2_action]

        self.k.vehicle.apply_acceleration(ids, actions)

    def compute_reward(self, rl_actions, **kwargs):

        reward_v1 = -abs((self.k.vehicle.get_speed(['v1_0'])[0]) - 0)
        reward_v2 = -abs((self.k.vehicle.get_speed(['v2_0'])[0]) - 3)
        
        return {'v1': reward_v1, 'v2': reward_v2}

    def get_state(self, **kwargs):

        state_v1 = self.k.vehicle.get_speed(['v1_0'])
        state_v2 = self.k.vehicle.get_speed(['v2_0'])
        
        return {'v1': state_v1, 'v2': state_v2}


class ConstantSpeedEnv5V(CustomAccelEnv, MultiEnv):

    def _apply_rl_actions(self, rl_actions):

        v1_action = rl_actions['v1']
        v2_action = rl_actions['v2']
        v3_action = rl_actions['v3']
        v4_action = rl_actions['v4']
        v5_action = rl_actions['v5']
        
        ids = ['v1_0', 'v2_0', 'v3_0', 'v4_0', 'v5_0']
        actions = [v1_action, v2_action, v3_action, v4_action, v5_action]

        self.k.vehicle.apply_acceleration(ids, actions)

    def compute_reward(self, rl_actions, **kwargs):

        reward_v1 = -abs((self.k.vehicle.get_speed(['v1_0'])[0]) - 0)
        reward_v2 = -abs((self.k.vehicle.get_speed(['v2_0'])[0]) - 1)
        reward_v3 = -abs((self.k.vehicle.get_speed(['v3_0'])[0]) - 2)
        reward_v4 = -abs((self.k.vehicle.get_speed(['v4_0'])[0]) - 3)
        reward_v5 = -abs((self.k.vehicle.get_speed(['v5_0'])[0]) - 4)

        return {'v1': reward_v1, 'v2': reward_v2, 'v3': reward_v3, 'v4': reward_v4, 'v5': reward_v5}

    def get_state(self, **kwargs):

        state_v1 = self.k.vehicle.get_speed(['v1_0'])
        state_v2 = self.k.vehicle.get_speed(['v2_0'])
        state_v3 = self.k.vehicle.get_speed(['v3_0'])
        state_v4 = self.k.vehicle.get_speed(['v4_0'])
        state_v5 = self.k.vehicle.get_speed(['v5_0'])

        return {'v1': state_v1, 'v2': state_v2, 'v3': state_v3, 'v4': state_v4, 'v5': state_v5,}


class ConstantSpeedEnv10V(CustomAccelEnv, MultiEnv):

    def _apply_rl_actions(self, rl_actions):

        v1_action = rl_actions['v1']
        v2_action = rl_actions['v2']
        v3_action = rl_actions['v3']
        v4_action = rl_actions['v4']
        v5_action = rl_actions['v5']
        v6_action = rl_actions['v6']
        v7_action = rl_actions['v7']
        v8_action = rl_actions['v8']
        v9_action = rl_actions['v9']
        v10_action = rl_actions['v10']
        
        ids = ['v1_0', 'v2_0', 'v3_0', 'v4_0', 'v5_0', 'v6_0', 'v7_0', 'v8_0', 'v9_0', 'v10_0']
        actions = [v1_action, v2_action, v3_action, v4_action, v5_action, v6_action, v7_action, v8_action, v9_action, v10_action]

        self.k.vehicle.apply_acceleration(ids, actions)

    def compute_reward(self, rl_actions, **kwargs):

        reward_v1 = -abs((self.k.vehicle.get_speed(['v1_0'])[0]) - 0)
        reward_v2 = -abs((self.k.vehicle.get_speed(['v2_0'])[0]) - 1)
        reward_v3 = -abs((self.k.vehicle.get_speed(['v3_0'])[0]) - 2)
        reward_v4 = -abs((self.k.vehicle.get_speed(['v4_0'])[0]) - 3)
        reward_v5 = -abs((self.k.vehicle.get_speed(['v5_0'])[0]) - 4)
        reward_v6 = -abs((self.k.vehicle.get_speed(['v6_0'])[0]) - 5)
        reward_v7 = -abs((self.k.vehicle.get_speed(['v7_0'])[0]) - 6)
        reward_v8 = -abs((self.k.vehicle.get_speed(['v8_0'])[0]) - 7)
        reward_v9 = -abs((self.k.vehicle.get_speed(['v9_0'])[0]) - 8)
        reward_v10 = -abs((self.k.vehicle.get_speed(['v10_0'])[0]) - 9)

        return {'v1': reward_v1, 'v2': reward_v2, 'v3': reward_v3, 'v4': reward_v4, 'v5': reward_v5, 'v6': reward_v6, 'v7': reward_v7, 'v8': reward_v8, 'v9': reward_v9, 'v10': reward_v10}

    def get_state(self, **kwargs):

        state_v1 = self.k.vehicle.get_speed(['v1_0'])
        state_v2 = self.k.vehicle.get_speed(['v2_0'])
        state_v3 = self.k.vehicle.get_speed(['v3_0'])
        state_v4 = self.k.vehicle.get_speed(['v4_0'])
        state_v5 = self.k.vehicle.get_speed(['v5_0'])
        state_v6 = self.k.vehicle.get_speed(['v6_0'])
        state_v7 = self.k.vehicle.get_speed(['v7_0'])
        state_v8 = self.k.vehicle.get_speed(['v8_0'])
        state_v9 = self.k.vehicle.get_speed(['v9_0'])
        state_v10 = self.k.vehicle.get_speed(['v10_0'])
        
        return {'v1': state_v1, 'v2': state_v2, 'v3': state_v3, 'v4': state_v4, 'v5': state_v5, 'v6': state_v6, 'v7': state_v7, 'v8': state_v8, 'v9': state_v9, 'v10': state_v10}
