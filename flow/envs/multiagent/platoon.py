"""Environment for training the acceleration behavior of vehicles in a ring."""
import numpy as np
from gym.spaces import Box

from flow.core import rewards
from flow.envs.ring.accel import AccelEnv
from flow.envs.multiagent.base import MultiEnv


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # desired velocity for all vehicles in the network, in m/s
    "target_velocity": 30,
    'sort_vehicles': True
}


class PlatoonEnv(AccelEnv, MultiEnv):
    """Adversarial multi-agent acceleration env.

    States
        The observation of both the AV and adversary agent consist of the
        velocities and absolute position of all vehicles in the network. This
        assumes a constant number of vehicles.

    Actions
        * AV: The action space of the AV agent consists of a vector of bounded
          accelerations for each autonomous vehicle. In order to ensure safety,
          these actions are further bounded by failsafes provided by the
          simulator at every time step.
        * Adversary: The action space of the adversary agent consists of a
          vector of perturbations to the accelerations issued by the AV agent.
          These are directly added to the original accelerations by the AV
          agent.

    Rewards
        * AV: The reward for the AV agent is equal to the mean speed of all
          vehicles in the network.
        * Adversary: The adversary receives a reward equal to the negative
          reward issued to the AV agent.

    Termination
        A rollout is terminated if the time horizon is reached or if two
        vehicles collide into one another.
    """

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

        self.k.vehicle.apply_acceleration(ids, rl_actions)

    def compute_reward(self, rl_actions, **kwargs):
        """Compute rewards for agents.
        """
        reward_follower0 = -abs((self.k.vehicle.get_speed(['follower0_0'])[0]) - 0)
        reward_follower1 = -abs((self.k.vehicle.get_speed(['follower1_0'])[0]) - 1)
        reward_follower2 = -abs((self.k.vehicle.get_speed(['follower2_0'])[0]) - 2)
        reward_follower3 = -abs((self.k.vehicle.get_speed(['follower3_0'])[0]) - 3)
        reward_follower4 = -abs((self.k.vehicle.get_speed(['follower4_0'])[0]) - 4)

        rewards = {'follower0': reward_follower0,
                   'follower1': reward_follower1,
                   'follower2': reward_follower2,
                   'follower3': reward_follower3,
                   'follower4': reward_follower4
                   }

        
        return rewards

    def get_state(self, **kwargs):
        """See class definition for the state.

        The adversary state and the agent state are identical.
        """
        state_follower0 = self.k.vehicle.get_speed(['follower0_0'])
        state_follower1 = self.k.vehicle.get_speed(['follower1_0'])
        state_follower2 = self.k.vehicle.get_speed(['follower2_0'])
        state_follower3 = self.k.vehicle.get_speed(['follower3_0'])
        state_follower4 = self.k.vehicle.get_speed(['follower4_0'])
        
        states = {  'follower0': state_follower0,
                    'follower1': state_follower1,
                    'follower2': state_follower2,
                    'follower3': state_follower3,
                    'follower4': state_follower4
                   }
        return states


