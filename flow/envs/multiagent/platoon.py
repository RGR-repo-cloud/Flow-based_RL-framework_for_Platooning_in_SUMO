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

        self.veh_ids = ['leader_0',
                        'follower0_0',
                        'follower1_0',
                        'follower2_0',
                        'follower3_0',
                        'follower4_0'
                        ]

    @property
    def action_space(self):
        """See class definition."""
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1, ),
            dtype=np.float32)


    def _apply_rl_actions(self, rl_actions):
        """See class definition."""

        action_follower0 = rl_actions[self.veh_ids[1]]
        action_follower1 = rl_actions[self.veh_ids[2]]
        action_follower2 = rl_actions[self.veh_ids[3]]
        action_follower3 = rl_actions[self.veh_ids[4]]
        action_follower4 = rl_actions[self.veh_ids[5]]
        
        rl_actions = [  action_follower0,
                        action_follower1,
                        action_follower2,
                        action_follower3,
                        action_follower4
                     ]
        
        # stored accelerations of RL vehicles must be updated manually
        for i, veh_id in enumerate(self.veh_ids[1:]):
            self.k.vehicle.update_accel(veh_id, rl_actions[i][0], noise=False, failsafe=False)

        self.k.vehicle.apply_acceleration(self.veh_ids[1:], rl_actions, smooth=False)


class UnilateralPlatoonEnv(PlatoonEnv):


    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)


    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity']
        return Box(
            low=-1000000, ###########should be made reasonable
            high=1000000, ###########should be made reasonable
            shape=(2, ), ##########unilateral
            dtype=np.float32)
    

    def compute_reward(self, rl_actions, **kwargs):
        """Compute rewards for agents.
        """
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])
        
        for headway in headways:
            if headway < 0 and not self.k.simulation.check_collision():
                print("!!!!negative headway")
                print(self.time_counter)
                print(headways)
                raise Exception

        # in case of a collision
        headways = [(headway if headway >= 0 else 0) for headway in headways]

        
        reward_follower0 = self.reward_function(headway=headways[0])
        reward_follower1 = self.reward_function(headway=headways[1])
        reward_follower2 = self.reward_function(headway=headways[2])
        reward_follower3 = self.reward_function(headway=headways[3])
        reward_follower4 = self.reward_function(headway=headways[4])

        if self.k.simulation.check_collision():

            reward_follower0 *= (self.env_params.horizon - self.time_counter)
            reward_follower1 *= (self.env_params.horizon - self.time_counter)
            reward_follower2 *= (self.env_params.horizon - self.time_counter)
            reward_follower3 *= (self.env_params.horizon - self.time_counter)
            reward_follower4 *= (self.env_params.horizon - self.time_counter)
            
        rewards = {self.veh_ids[1]: reward_follower0,
                   self.veh_ids[2]: reward_follower1,
                   self.veh_ids[3]: reward_follower2,
                   self.veh_ids[4]: reward_follower3,
                   self.veh_ids[5]: reward_follower4
                   }

        
        return rewards

    def get_state(self, **kwargs):

        speeds = self.k.vehicle.get_speed(self.veh_ids)
        previous_speeds = self.k.vehicle.get_previous_speed(self.veh_ids)
        # checking for crashed vehicle
        accelerations = [(0 if not isinstance(self.k.vehicle.get_arrived_ids(), int) and veh_id in self.k.vehicle.get_arrived_ids()
                         else self.k.vehicle.get_accel(veh_id, noise=False, failsafe=False))
                        for veh_id in self.veh_ids]
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])
        
        for acceleration in accelerations:
            if self.time_counter is not 0 and abs(acceleration) > 3.1:
                print("!!!!!!!too high input accel")
                print(self.time_counter)
                print(accelerations)
                raise Exception

        for i, speed in enumerate(speeds):
            if speed < 0:
                print("!!!negative speed!!!")
                print(self.time_counter)
                print(speeds)
                raise Exception
            if abs(speed - previous_speeds[i]) > 0.31 and self.time_counter is not 0:
                print("!!!!!!!!!!!!!!too high accel!!!!!!!!!!!!!!!!!!!!")
                print(self.time_counter)
                print(speed - previous_speeds[i])
                print(accelerations)
                raise Exception
            if previous_speeds[i] is 0 and speed >= 0 and self.time_counter is not 0:
                print("!!!faulty emergency brake!!!")
                print(self.time_counter)
                print(speed)
                print(previous_speeds[i])
                raise Exception
        
        for headway in headways:
            if headway < 0 and not self.k.simulation.check_collision():
                print("!!!!negative headway")
                print(self.time_counter)
                print(headways)
                raise Exception
        
        # in case of a collision
        headways = [(headway if headway >= 0 else 0) for headway in headways]
        speeds = [(speed if speed >= 0 else previous_speeds[i]) for i, speed in enumerate(speeds)]

        state_follower0 = [-speeds[1] + speeds[0], headways[0]]
        state_follower1 = [-speeds[2] + speeds[1], headways[1]]
        state_follower2 = [-speeds[3] + speeds[2], headways[2]]
        state_follower3 = [-speeds[4] + speeds[3], headways[3]]
        state_follower4 = [-speeds[5] + speeds[4], headways[4]]
        
        states = {  self.veh_ids[1]: state_follower0,
                    self.veh_ids[2]: state_follower1,
                    self.veh_ids[3]: state_follower2,
                    self.veh_ids[4]: state_follower3,
                    self.veh_ids[5]: state_follower4
                   }
        
        return states
    

    def reward_function(self, headway):

        """
        if headway >= 30:
            return -abs(headway - 20)
        if headway >= 10:
            return -abs(headway - 20)
        if headway >= 5:
            return -abs(headway - 20)
        else:
            return -abs(pow(headway - 20, 2))
        """
        return -abs(headway - 20)


class BilateralPlatoonEnv(PlatoonEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)


    @property
    def observation_space(self):
        """See class definition."""
        self.obs_var_labels = ['Velocity']
        return Box(
            low=-1000000, ###########should be made reasonable
            high=1000000, ###########should be made reasonable
            shape=(7, ), ##########unilateral
            dtype=np.float32)
    

    def compute_reward(self, rl_actions, **kwargs):
        """Compute rewards for agents.
        """
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])

        reward_follower0 = self.reward_function(headway=headways[0])
        reward_follower1 = self.reward_function(headway=headways[1])
        reward_follower2 = self.reward_function(headway=headways[2])
        reward_follower3 = self.reward_function(headway=headways[3])
        reward_follower4 = self.reward_function(headway=headways[4])
        
        rewards = {self.veh_ids[1]: reward_follower0,
                   self.veh_ids[2]: reward_follower1,
                   self.veh_ids[3]: reward_follower2,
                   self.veh_ids[4]: reward_follower3,
                   self.veh_ids[5]: reward_follower4
                   }

        
        return rewards

    def get_state(self, **kwargs):

        speeds = self.k.vehicle.get_speed(self.veh_ids)
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])

        # if collisions occur a false headway is given
        headways = [(headway if headway >= 0 else 0) for headway in headways]


        state_follower0 = [-speeds[1] + speeds[0], -speeds[1] + speeds[2], headways[0], headways[1], accelerations[0], accelerations[1], accelerations[2]]
        state_follower1 = [-speeds[2] + speeds[1], -speeds[2] + speeds[3], headways[1], headways[2], accelerations[1], accelerations[2], accelerations[3]]
        state_follower2 = [-speeds[3] + speeds[2], -speeds[3] + speeds[4], headways[2], headways[3], accelerations[2], accelerations[3], accelerations[4]]
        state_follower3 = [-speeds[4] + speeds[3], -speeds[4] + speeds[5], headways[3], headways[4], accelerations[3], accelerations[4], accelerations[5]]
        state_follower4 = [-speeds[5] + speeds[4], 0, headways[4], 0, accelerations[4], accelerations[5], 0] #should be different dimension
        
        states = {  self.veh_ids[1]: state_follower0,
                    self.veh_ids[2]: state_follower1,
                    self.veh_ids[3]: state_follower2,
                    self.veh_ids[4]: state_follower3,
                    self.veh_ids[5]: state_follower4
                   }
        
        return states
    

    def reward_function(self, headway):


        if headway >= 30:
            return -10*abs(headway - 20)
        if headway >= 10:
            return -5*abs(headway - 20)
        if headway >= 5:
            return -abs(headway - 20)
        else:
            return -abs(pow(headway - 20, 2))


