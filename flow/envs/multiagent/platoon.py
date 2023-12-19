"""Environment for training the acceleration behavior of vehicles in a ring."""
import numpy as np
from gym.spaces import Box

from flow.core import rewards
from flow.envs.multiagent.base import MultiEnv


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 30
}


class PlatoonEnv(MultiEnv):

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
            low=-1000000, ###########should be made reasonable
            high=1000000, ###########should be made reasonable
            shape=(2, ), ##########unilateral
            dtype=np.float32)


    def _apply_rl_actions(self, rl_actions):
        """See class definition."""

        action_follower0 = rl_actions['follower0']
        action_follower1 = rl_actions['follower1']
        action_follower2 = rl_actions['follower2']
        action_follower3 = rl_actions['follower3']
        action_follower4 = rl_actions['follower4']
        
        ids = [ 'follower0_0',
                'follower1_0',
                'follower2_0',
                'follower3_0',
                'follower4_0'
                ]
        
        rl_actions = [  action_follower0,
                        action_follower1,
                        action_follower2,
                        action_follower3,
                        action_follower4
                     ]

        self.k.vehicle.apply_acceleration(ids, rl_actions, smooth=True)

    def compute_reward(self, rl_actions, **kwargs):
        """Compute rewards for agents.
        """
        headways = self.k.vehicle.get_headway([ 'follower0_0',
                                                'follower1_0',
                                                'follower2_0',
                                                'follower3_0',
                                                'follower4_0'])

        reward_follower0 = self.reward_function(headway=headways[0], tailway=headways[1])
        reward_follower1 = self.reward_function(headway=headways[1], tailway=headways[2])
        reward_follower2 = self.reward_function(headway=headways[2], tailway=headways[3])
        reward_follower3 = self.reward_function(headway=headways[3], tailway=headways[4])
        reward_follower4 = self.reward_function(headway=headways[4], tailway=20)

        rewards = {'follower0': reward_follower0,
                   'follower1': reward_follower1,
                   'follower2': reward_follower2,
                   'follower3': reward_follower3,
                   'follower4': reward_follower4
                   }

        
        return rewards

    def get_state(self, **kwargs):

        speeds = self.k.vehicle.get_speed([ 'leader_0',
                                            'follower0_0',
                                            'follower1_0',
                                            'follower2_0',
                                            'follower3_0',
                                            'follower4_0'
                                            ])
        headways = self.k.vehicle.get_headway([ 'follower0_0',
                                                'follower1_0',
                                                'follower2_0',
                                                'follower3_0',
                                                'follower4_0'
                                                ])

        state_follower0 = [-speeds[1] + speeds[0], headways[0]]
        state_follower1 = [-speeds[2] + speeds[1], headways[1]]
        state_follower2 = [-speeds[3] + speeds[2], headways[2]]
        state_follower3 = [-speeds[4] + speeds[3], headways[3]]
        state_follower4 = [-speeds[5] + speeds[4], headways[4]]
        
        states = {  'follower0': state_follower0,
                    'follower1': state_follower1,
                    'follower2': state_follower2,
                    'follower3': state_follower3,
                    'follower4': state_follower4
                   }
        return states
    

    def reward_function(self, headway, tailway):


        if headway >= 30:
            return -10*abs(headway - 20)
        if headway >= 10:
            return -5*abs(headway - 20)
        if headway >= 5:
            return -abs(headway - 20)
        else:
            return -abs(pow(headway - 20, 2))


