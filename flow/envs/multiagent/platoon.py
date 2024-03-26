import numpy as np
from gym.spaces import Box, Dict

from flow.core import rewards
from flow.envs.multiagent.base import MultiEnv
from queue import Queue
import math


ADDITIONAL_ENV_PARAMS = {
    # maximum acceleration for autonomous vehicles, in m/s^2
    "max_accel": 3,
    # maximum deceleration for autonomous vehicles, in m/s^2
    "max_decel": 3,
    # number of scenarios
    "num_scenarios": 1,
    # size of the state's time frame
    "state_time_frame": 1,
    # use a modified variant of the original reward function for training
    "modified_reward_function": False
    }


def reward_function_bilateral(headway_front, headway_rear, speed_front, speed_self, speed_rear, accel, previous_accel,control_input, time_gap, standstill_distance):

    max_gap_error = 15
    max_speed_error = 10
    max_accel = 3
    reward_scale = 0.01

    gap_error_front = headway_front - (standstill_distance + speed_self * time_gap)
    gap_error_rear = headway_rear - (standstill_distance + speed_rear * time_gap)
    gap_error_rear = gap_error_rear if gap_error_rear < 0 else 0
    speed_error_front = speed_front - speed_self
    speed_error_rear = speed_self - speed_rear
    speed_error_rear = speed_error_rear if speed_error_rear < 0 else 0
    jerk = accel - previous_accel

    normed_gap_error_front = abs(gap_error_front / max_gap_error)
    normed_gap_error_rear = abs(gap_error_rear / max_gap_error)
    normed_speed_error_front = abs(speed_error_front / max_speed_error)
    normed_speed_error_rear = abs(speed_error_rear / max_speed_error)
    normed_input_penalty = abs((control_input / max_accel))
    normed_jerk = abs(jerk / (2 * max_accel))

    weight_a = 0.1
    weight_b = 0.1
    weight_c = 0.2
    weight_d = 0.4
    weight_e = 0.04

    normed_reward = -(normed_gap_error_front + 
                    (weight_a * normed_speed_error_front) + 
                    (weight_b * normed_input_penalty) + 
                    (weight_c * normed_jerk) +
                    (weight_d * normed_gap_error_rear) +
                    (weight_e * normed_speed_error_rear)
                    )
    sqr_reward = -(pow(gap_error_front, 2) + 
                    (weight_a * pow(speed_error_front, 2)) + 
                    (weight_b * pow(control_input, 2)) +
                    (weight_c * pow(jerk, 2)) +
                    (weight_d * pow(gap_error_rear, 2)) +
                    (weight_e * pow(speed_error_rear, 2)))

    epsilon = -0.4483

    if normed_reward < epsilon:
        reward = normed_reward
    else:
        reward = reward_scale * sqr_reward
    
    return reward
    

def reward_function_unilateral(headway, speed_front, speed_self, accel, previous_accel, control_input, time_gap, standstill_distance):
    
    max_gap_error = 15
    max_speed_error = 10
    max_accel = 3
    reward_scale = 0.01
    time_step = 0.1

    gap_error = headway - (standstill_distance + speed_self * time_gap)
    speed_error = speed_front - speed_self
    jerk = (accel - previous_accel) / time_step

    normed_gap_error = abs(gap_error / max_gap_error)
    normed_speed_error = abs(speed_error / max_speed_error)
    normed_input_penalty = abs((control_input / max_accel))
    normed_jerk = abs(jerk / (2 * max_accel / time_step))


    weight_a = 0.1
    weight_b = 0.1
    weight_c = 0.2

    normed_reward = -(normed_gap_error + 
                    (weight_a * normed_speed_error) + 
                    (weight_b * normed_input_penalty) + 
                    (weight_c * normed_jerk))
    sqr_reward = -(pow(gap_error, 2) + 
                    (weight_a * pow(speed_error, 2)) + 
                    (weight_b * pow(control_input, 2)) +
                    (weight_c * pow(jerk * time_step, 2)))

    epsilon = -0.4483

    if normed_reward < epsilon:
        reward = normed_reward
    else:
        reward = reward_scale * sqr_reward
    
    return reward, gap_error, speed_error, jerk


def mod_reward_function(reward, crashed):

    if crashed:
        return 0
        
    if reward >= -1:
        return reward + 2
        
    return custom_sigmoid(reward + 1)
        


def custom_sigmoid(x):
    return 2 / (1 + math.exp(-x))



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
        
        self.previous_accels = {}
        for veh_id in self.veh_ids[1:]:
            self.previous_accels[veh_id] = 0

        self.state_frame_size = self.env_params.additional_params['state_time_frame']
        self.previous_states = {}
        self.state_frame = {}
        self.last_time = self.time_counter

        self.eval_state_dict = {}
        for agent_id in self.veh_ids[1:]:
            self.eval_state_dict[agent_id] = {}
        
        self.eval_reward_dict = {}
        for agent_id in self.veh_ids[1:]:
            self.eval_reward_dict[agent_id] = {}

        self.eval_leader_dict = {}


    @property
    def action_space(self):
        return Box(
            low=-abs(self.env_params.additional_params['max_decel']),
            high=self.env_params.additional_params['max_accel'],
            shape=(1, ),
            dtype=np.float32)


    def _apply_rl_actions(self, rl_actions):

        for veh_id in self.veh_ids[1:]:
            self.previous_accels[veh_id] = self.k.vehicle.get_accel(veh_id, noise=False, failsafe=False)
            if self.previous_accels[veh_id] is None:
                self.previous_accels[veh_id] = 0


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


    def add_previous_state(self, state):
        
        for veh_id in self.veh_ids[1:]:
            del self.previous_states[veh_id][-1]
            self.previous_states[veh_id].insert(0, state[veh_id])
            assert len(self.previous_states[veh_id]) == self.state_frame_size


    def init_state_frame(self, state):
        
        self.previous_states = {}
        for veh_id in self.veh_ids[1:]:
            self.previous_states[veh_id] = []
            for _ in range(self.state_frame_size):
                self.previous_states[veh_id].append(state[veh_id])
            assert len(self.previous_states[veh_id]) == self.state_frame_size

    
    def create_state_frame(self):

        state_frame = {}
        for veh_id in self.veh_ids[1:]:
            veh_state_frame = []
            for i in range(self.state_frame_size):
                veh_state_frame += self.previous_states[veh_id][i]
            state_frame[veh_id] = veh_state_frame
        
        return state_frame
    

    def add_to_eval_state_data(self, key, values):
        
        current_scenario = self.scenario_tracker

        for id, agent_id in enumerate(self.veh_ids[1:]):
            if current_scenario not in self.eval_state_dict[agent_id]:
                self.eval_state_dict[agent_id][current_scenario] = {}
            if key not in self.eval_state_dict[agent_id][current_scenario].keys():
                self.eval_state_dict[agent_id][current_scenario][key] = []
            self.eval_state_dict[agent_id][current_scenario][key].append(values[id])

    def add_to_eval_reward_data(self, key, values):
        
        current_scenario = self.scenario_tracker

        for id, agent_id in enumerate(self.veh_ids[1:]):
            if current_scenario not in self.eval_reward_dict[agent_id]:
                self.eval_reward_dict[agent_id][current_scenario] = {}
            if key not in self.eval_reward_dict[agent_id][current_scenario].keys():
                self.eval_reward_dict[agent_id][current_scenario][key] = []
            self.eval_reward_dict[agent_id][current_scenario][key].append(values[id])

    def add_to_eval_leader(self, key, value):
        
        current_scenario = self.scenario_tracker
        if current_scenario not in self.eval_leader_dict:
            self.eval_leader_dict[current_scenario] = {}
        if key not in self.eval_leader_dict[current_scenario].keys():
            self.eval_leader_dict[current_scenario][key] = []
        self.eval_leader_dict[current_scenario][key].append(value)


class UnilateralPlatoonEnv(PlatoonEnv):


    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.standstill_distance = 2
        self.time_gap = 1


    @property
    def observation_space(self):
        obs_space = {}
        for veh_id in self.veh_ids[1:]:
            obs_space[veh_id] = Box(
                                low=-1000000, # an arbitrary high enough number
                                high=1000000, # an arbitrary high enough number
                                shape=(4 * self.state_frame_size, ), # unilateral
                                dtype=np.float32)
        return Dict(obs_space)
    

    def compute_reward(self, rl_actions, **kwargs):

        # retrieve state information
        accelerations = [(0 if not isinstance(self.k.vehicle.get_arrived_ids(), int) and veh_id in self.k.vehicle.get_arrived_ids()
                         else self.k.vehicle.get_accel(veh_id, noise=False, failsafe=False))
                        for veh_id in self.veh_ids]
        
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])
        speeds = self.k.vehicle.get_speed(self.veh_ids)
        

        # in case of a collision
        headways = [(headway if headway >= 0 else 0) for headway in headways]

        # identify crashed vehicles
        crashed = {}
        for veh_id in self.veh_ids[1:]:
            crashed[veh_id] = veh_id in self.k.vehicle.get_arrived_rl_ids(self.env_params.sims_per_step)

        if self.mode == 'train':
            # calculate training rewards
            reward_follower0, gap_error0, speed_error0, jerk0  = reward_function_unilateral(headways[0], speeds[0], speeds[1], accelerations[1], self.previous_accels[self.veh_ids[1]], self.time_gap, self.standstill_distance)
            reward_follower1, gap_error1, speed_error1, jerk1 = reward_function_unilateral(headways[1], speeds[1], speeds[2], accelerations[2], self.previous_accels[self.veh_ids[2]], self.time_gap, self.standstill_distance)
            reward_follower2, gap_error2, speed_error2, jerk2 = reward_function_unilateral(headways[2], speeds[2], speeds[3], accelerations[3], self.previous_accels[self.veh_ids[3]], self.time_gap, self.standstill_distance)
            reward_follower3, gap_error3, speed_error3, jerk3 = reward_function_unilateral(headways[3], speeds[3], speeds[4], accelerations[4], self.previous_accels[self.veh_ids[4]], self.time_gap, self.standstill_distance)
            reward_follower4, gap_error4, speed_error4, jerk4 = reward_function_unilateral(headways[4], speeds[4], speeds[5], accelerations[5], self.previous_accels[self.veh_ids[5]], self.time_gap, self.standstill_distance)
            
            if self.env_params.additional_params['modified_reward_function']:
                reward_follower0 = mod_reward_function(reward_follower0, crashed[self.veh_ids[1]])
                reward_follower1 = mod_reward_function(reward_follower1, crashed[self.veh_ids[2]])
                reward_follower2 = mod_reward_function(reward_follower2, crashed[self.veh_ids[3]])
                reward_follower3 = mod_reward_function(reward_follower3, crashed[self.veh_ids[4]])
                reward_follower4 = mod_reward_function(reward_follower4, crashed[self.veh_ids[5]])

        elif self.mode == 'eval':
            # calculate evaluation rewards
            reward_follower0, gap_error0, speed_error0, jerk0  = reward_function_unilateral(headways[0], speeds[0], speeds[1], accelerations[1], self.previous_accels[self.veh_ids[1]], self.time_gap, self.standstill_distance)
            reward_follower1, gap_error1, speed_error1, jerk1 = reward_function_unilateral(headways[1], speeds[1], speeds[2], accelerations[2], self.previous_accels[self.veh_ids[2]], self.time_gap, self.standstill_distance)
            reward_follower2, gap_error2, speed_error2, jerk2 = reward_function_unilateral(headways[2], speeds[2], speeds[3], accelerations[3], self.previous_accels[self.veh_ids[3]], self.time_gap, self.standstill_distance)
            reward_follower3, gap_error3, speed_error3, jerk3 = reward_function_unilateral(headways[3], speeds[3], speeds[4], accelerations[4], self.previous_accels[self.veh_ids[4]], self.time_gap, self.standstill_distance)
            reward_follower4, gap_error4, speed_error4, jerk4 = reward_function_unilateral(headways[4], speeds[4], speeds[5], accelerations[5], self.previous_accels[self.veh_ids[5]], self.time_gap, self.standstill_distance)
        else:
            raise Exception("no valid reward mode")

            
        rewards = {self.veh_ids[1]: reward_follower0,
                   self.veh_ids[2]: reward_follower1,
                   self.veh_ids[3]: reward_follower2,
                   self.veh_ids[4]: reward_follower3,
                   self.veh_ids[5]: reward_follower4
                   }
        
        # log evaluation data
        if self.env_params.evaluate:
            rl_actions_list = []
            for veh_id in self.veh_ids[1:]:
                rl_actions_list.append(rl_actions[veh_id][0])
            
            self.add_to_eval_reward_data('step', [self.time_counter]*len(self.veh_ids[1:]))
            self.add_to_eval_reward_data('input', rl_actions_list)
            self.add_to_eval_reward_data('gap_error', [gap_error0, gap_error1, gap_error2, gap_error3, gap_error4])
            self.add_to_eval_reward_data('speed_error', [speed_error0, speed_error1, speed_error2, speed_error3, speed_error4])
            self.add_to_eval_reward_data('jerk', [jerk0, jerk1, jerk2, jerk3, jerk4])
            self.add_to_eval_reward_data('reward', [reward_follower0, reward_follower1, reward_follower2, reward_follower3, reward_follower4])


        return rewards

    def get_state(self, **kwargs):

        # retrieve state information
        speeds = self.k.vehicle.get_speed(self.veh_ids)
        previous_speeds = self.k.vehicle.get_previous_speed(self.veh_ids)
        accelerations = [(0 if not isinstance(self.k.vehicle.get_arrived_ids(), int) and veh_id in self.k.vehicle.get_arrived_ids()
                         else self.k.vehicle.get_accel(veh_id, noise=False, failsafe=False))
                        for veh_id in self.veh_ids]
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])
        
        
        # in case of a collision
        headways = [(headway if headway >= 0 else 0) for headway in headways]
        speeds = [(speed if speed >= 0 else previous_speeds[i]) for i, speed in enumerate(speeds)]
        gap_errors = [((self.standstill_distance + speeds[i+1] * self.time_gap) - headways[i]) for i in range(len(self.veh_ids[1:]))]

        # accelerations are not updated at the start
        accelerations = [0 if accelerations[i] is None else accelerations[i] for i in range(len(self.veh_ids))]

        # log state information
        if self.env_params.evaluate:
            if self.last_time == self.time_counter:
                self.last_time = -1
            else:
                self.add_to_eval_state_data('step', [self.time_counter]*len(self.veh_ids[1:]))
                self.add_to_eval_state_data('speed', speeds[1:])
                self.add_to_eval_state_data('accel', accelerations[1:])
                self.add_to_eval_state_data('headway', headways)

                self.add_to_eval_leader('step', self.time_counter)
                self.add_to_eval_leader('accel', self.k.vehicle.get_realized_accel('leader_0'))
                self.add_to_eval_leader('speed', speeds[0])
                
                self.last_time = self.time_counter
            

        # current states
        states = {
            self.veh_ids[1]: [-speeds[1] + speeds[0], gap_errors[0], accelerations[1], self.k.vehicle.get_realized_accel('leader_0')],
            self.veh_ids[2]: [-speeds[2] + speeds[1], gap_errors[1], accelerations[2], accelerations[1]],
            self.veh_ids[3]: [-speeds[3] + speeds[2], gap_errors[2], accelerations[3], accelerations[2]],
            self.veh_ids[4]: [-speeds[4] + speeds[3], gap_errors[3], accelerations[4], accelerations[3]],
            self.veh_ids[5]: [-speeds[5] + speeds[4], gap_errors[4], accelerations[5], accelerations[4]]
        }
        
        # time frame of states
        if self.state_frame_size > 1:
            if self.time_counter == 0:
                self.init_state_frame(states)
            else:
                self.add_previous_state(states)
            states = self.create_state_frame()


        return states


class BilateralPlatoonEnv(PlatoonEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.standstill_distance = 2
        self.time_gap = 1


    @property
    def observation_space(self):
        obs_space = {}
        for veh_id in self.veh_ids[1:-1]:
            obs_space[veh_id] = Box(
                                low=-1000000, # an arbitrary high enough number
                                high=1000000, # an arbitrary high enough number
                                shape=(7 * self.state_frame_size, ), # bilateral
                                dtype=np.float32)
        # last follower has no rear vehicle
        obs_space[self.veh_ids[-1]] = Box(
                                low=-1000000, # an arbitrary high enough number
                                high=1000000, # an arbitrary high enough number
                                shape=(4 * self.state_frame_size, ), # unilateral
                                dtype=np.float32)
        
        return Dict(obs_space)
    

    def compute_reward(self, rl_actions, **kwargs):

        # retrieve state information
        accelerations = [(0 if not isinstance(self.k.vehicle.get_arrived_ids(), int) and veh_id in self.k.vehicle.get_arrived_ids()
                         else self.k.vehicle.get_accel(veh_id, noise=False, failsafe=False))
                        for veh_id in self.veh_ids]
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])
        speeds = self.k.vehicle.get_speed(self.veh_ids)

        # in case of a collision
        headways = [(headway if headway >= 0 else 0) for headway in headways]

        # identify crashed vehicles
        crashed = {}
        for veh_id in self.veh_ids[1:]:
            crashed[veh_id] = veh_id in self.k.vehicle.get_arrived_rl_ids(self.env_params.sims_per_step)


        if self.mode == 'train':
            # compute training rewards
            reward_follower0 = reward_function_bilateral(headways[0], headways[1], speeds[0], speeds[1], speeds[2], accelerations[1], self.previous_accels[self.veh_ids[1]], self.time_gap, self.standstill_distance)
            reward_follower1 = reward_function_bilateral(headways[1], headways[2], speeds[1], speeds[2], speeds[3], accelerations[2], self.previous_accels[self.veh_ids[2]], self.time_gap, self.standstill_distance)
            reward_follower2 = reward_function_bilateral(headways[2], headways[3], speeds[2], speeds[3], speeds[4], accelerations[3], self.previous_accels[self.veh_ids[3]], self.time_gap, self.standstill_distance)
            reward_follower3 = reward_function_bilateral(headways[3], headways[4], speeds[3], speeds[4], speeds[5], accelerations[4], self.previous_accels[self.veh_ids[4]], self.time_gap, self.standstill_distance)
            reward_follower4,_,_,_ = reward_function_unilateral(headways[4], speeds[4], speeds[5], accelerations[5], self.previous_accels[self.veh_ids[5]], self.time_gap, self.standstill_distance)

            if self.env_params.additional_params['modified_reward_function']:
                reward_follower0 = mod_reward_function(reward_follower0, crashed[self.veh_ids[1]])
                reward_follower1 = mod_reward_function(reward_follower1, crashed[self.veh_ids[2]])
                reward_follower2 = mod_reward_function(reward_follower2, crashed[self.veh_ids[3]])
                reward_follower3 = mod_reward_function(reward_follower3, crashed[self.veh_ids[4]])
                reward_follower4 = mod_reward_function(reward_follower4, crashed[self.veh_ids[5]])
        
        elif self.mode == 'eval':
            # compute evaluation rewards
            reward_follower0, gap_error0, speed_error0, jerk0 = reward_function_unilateral(headways[0], speeds[0], speeds[1], accelerations[1], self.previous_accels[self.veh_ids[1]], self.time_gap, self.standstill_distance)
            reward_follower1, gap_error1, speed_error1, jerk1 = reward_function_unilateral(headways[1], speeds[1], speeds[2], accelerations[2], self.previous_accels[self.veh_ids[2]], self.time_gap, self.standstill_distance)
            reward_follower2, gap_error2, speed_error2, jerk2 = reward_function_unilateral(headways[2], speeds[2], speeds[3], accelerations[3], self.previous_accels[self.veh_ids[3]], self.time_gap, self.standstill_distance)
            reward_follower3, gap_error3, speed_error3, jerk3 = reward_function_unilateral(headways[3], speeds[3], speeds[4], accelerations[4], self.previous_accels[self.veh_ids[4]], self.time_gap, self.standstill_distance)
            reward_follower4, gap_error4, speed_error4, jerk4 = reward_function_unilateral(headways[4], speeds[4], speeds[5], accelerations[5], self.previous_accels[self.veh_ids[5]], self.time_gap, self.standstill_distance)
        
        else:
            raise Exception("no valid reward mode")
    
            
        rewards = {self.veh_ids[1]: reward_follower0,
                   self.veh_ids[2]: reward_follower1,
                   self.veh_ids[3]: reward_follower2,
                   self.veh_ids[4]: reward_follower3,
                   self.veh_ids[5]: reward_follower4
                   }

        # log evaluation data
        if self.env_params.evaluate:
            rl_actions_list = []
            for veh_id in self.veh_ids[1:]:
                rl_actions_list.append(rl_actions[veh_id][0])
            
            self.add_to_eval_reward_data('step', [self.time_counter]*len(self.veh_ids[1:]))
            self.add_to_eval_reward_data('input', rl_actions_list)
            self.add_to_eval_reward_data('gap_error', [gap_error0, gap_error1, gap_error2, gap_error3, gap_error4])
            self.add_to_eval_reward_data('speed_error', [speed_error0, speed_error1, speed_error2, speed_error3, speed_error4])
            self.add_to_eval_reward_data('jerk', [jerk0, jerk1, jerk2, jerk3, jerk4])
            self.add_to_eval_reward_data('reward', [reward_follower0, reward_follower1, reward_follower2, reward_follower3, reward_follower4])
        

        return rewards


    def get_state(self, **kwargs):

        # retrieve state information
        speeds = self.k.vehicle.get_speed(self.veh_ids)
        previous_speeds = self.k.vehicle.get_previous_speed(self.veh_ids)
        accelerations = [(0 if not isinstance(self.k.vehicle.get_arrived_ids(), int) and veh_id in self.k.vehicle.get_arrived_ids()
                         else self.k.vehicle.get_accel(veh_id, noise=False, failsafe=False))
                        for veh_id in self.veh_ids]
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])
        
        # in case of a collision
        headways = [(headway if headway >= 0 else 0) for headway in headways]
        speeds = [(speed if speed >= 0 else previous_speeds[i]) for i, speed in enumerate(speeds)]
        gap_errors = [((self.standstill_distance + speeds[i+1] * self.time_gap) - headways[i]) for i in range(len(self.veh_ids[1:]))]

        # accelerations are not updated at the start
        accelerations = [0 if accelerations[i] is None else accelerations[i] for i in range(len(self.veh_ids))]

        # log state information
        if self.env_params.evaluate:
            if self.last_time == self.time_counter:
                self.last_time = -1
            else:
                self.add_to_eval_state_data('step', [self.time_counter]*len(self.veh_ids[1:]))
                self.add_to_eval_state_data('speed', speeds[1:])
                self.add_to_eval_state_data('accel', accelerations[1:])
                self.add_to_eval_state_data('headway', headways)

                self.add_to_eval_leader('step', self.time_counter)
                self.add_to_eval_leader('accel', self.k.vehicle.get_realized_accel('leader_0'))
                self.add_to_eval_leader('speed', speeds[0])
                
                self.last_time = self.time_counter

        # current states
        states = {
            self.veh_ids[1]: [-speeds[1] + speeds[0], -speeds[1] + speeds[2], gap_errors[0], gap_errors[1], accelerations[1], self.k.vehicle.get_realized_accel('leader_0'), accelerations[2]], 
            self.veh_ids[2]: [-speeds[2] + speeds[1], -speeds[2] + speeds[3], gap_errors[1], gap_errors[2], accelerations[2], accelerations[1], accelerations[3]],
            self.veh_ids[3]: [-speeds[3] + speeds[2], -speeds[3] + speeds[4], gap_errors[2], gap_errors[3], accelerations[3], accelerations[2], accelerations[4]],
            self.veh_ids[4]: [-speeds[4] + speeds[3], -speeds[4] + speeds[5], gap_errors[3], gap_errors[4], accelerations[4], accelerations[3], accelerations[5]],
            self.veh_ids[5]: [-speeds[5] + speeds[4], gap_errors[4], accelerations[5], accelerations[4]] # last follower has no successor
        }
        
        # time frame of states
        if self.state_frame_size > 1:
            if self.time_counter == 0:
                self.init_state_frame(states)
            else:
                self.add_previous_state(states)
            states = self.create_state_frame()


        return states


class FlatbedEnv(PlatoonEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.standstill_distance = 2
        self.time_gap = 1

    @property
    def observation_space(self):
        return Box(
            low=-1000000, # an arbitrary high enough number
            high=1000000, # an arbitrary high enough number
            shape=(5, ),
            dtype=np.float32)
    

    def compute_reward(self, rl_actions, **kwargs):

        # retrieve state information
        accelerations = [(0 if not isinstance(self.k.vehicle.get_arrived_ids(), int) and veh_id in self.k.vehicle.get_arrived_ids()
                         else self.k.vehicle.get_accel(veh_id, noise=False, failsafe=False))
                        for veh_id in self.veh_ids]
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])
        speeds = self.k.vehicle.get_speed(self.veh_ids)

        # identify crashed vehicles
        crashed = {}
        for veh_id in self.veh_ids[1:]:
            crashed[veh_id] = veh_id in self.k.vehicle.get_arrived_rl_ids(self.env_params.sims_per_step)

        # compute rewards
        reward_follower0, gap_error0, speed_error0, jerk0 = reward_function_unilateral(headways[0], speeds[0], speeds[1], accelerations[1], self.previous_accels[self.veh_ids[1]], self.time_gap, self.standstill_distance)
        reward_follower1, gap_error1, speed_error1, jerk1 = reward_function_unilateral(headways[1], speeds[1], speeds[2], accelerations[2], self.previous_accels[self.veh_ids[2]], self.time_gap, self.standstill_distance)
        reward_follower2, gap_error2, speed_error2, jerk2 = reward_function_unilateral(headways[2], speeds[2], speeds[3], accelerations[3], self.previous_accels[self.veh_ids[3]], self.time_gap, self.standstill_distance)
        reward_follower3, gap_error3, speed_error3, jerk3 = reward_function_unilateral(headways[3], speeds[3], speeds[4], accelerations[4], self.previous_accels[self.veh_ids[4]], self.time_gap, self.standstill_distance)
        reward_follower4, gap_error4, speed_error4, jerk4 = reward_function_unilateral(headways[4], speeds[4], speeds[5], accelerations[5], self.previous_accels[self.veh_ids[5]], self.time_gap, self.standstill_distance)

        # log evaluation data
        if self.env_params.evaluate:
            rl_actions_list = []
            for veh_id in self.veh_ids[1:]:
                rl_actions_list.append(rl_actions[veh_id][0])
            
            self.add_to_eval_reward_data('step', [self.time_counter]*len(self.veh_ids[1:]))
            self.add_to_eval_reward_data('input', rl_actions_list)
            self.add_to_eval_reward_data('gap_error', [gap_error0, gap_error1, gap_error2, gap_error3, gap_error4])
            self.add_to_eval_reward_data('speed_error', [speed_error0, speed_error1, speed_error2, speed_error3, speed_error4])
            self.add_to_eval_reward_data('jerk', [jerk0, jerk1, jerk2, jerk3, jerk4])

        rewards = {self.veh_ids[1]: reward_follower0,
                   self.veh_ids[2]: reward_follower1,
                   self.veh_ids[3]: reward_follower2,
                   self.veh_ids[4]: reward_follower3,
                   self.veh_ids[5]: reward_follower4
                   }

        return rewards
    

    def get_state(self, **kwargs):

        # retrieve state information
        speeds = self.k.vehicle.get_speed(self.veh_ids)
        accelerations = [(0 if not isinstance(self.k.vehicle.get_arrived_ids(), int) and veh_id in self.k.vehicle.get_arrived_ids()
                         else self.k.vehicle.get_accel(veh_id, noise=False, failsafe=False))
                        for veh_id in self.veh_ids]
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])

        # accelerations are not updated at the start
        accelerations = [0 if accelerations[i] is None else accelerations[i] for i in range(len(self.veh_ids))]

        # log state
        if self.env_params.evaluate:
            if self.last_time == self.time_counter:
                self.last_time = -1
            else:
                self.add_to_eval_state_data('step', [self.time_counter]*len(self.veh_ids[1:]))
                self.add_to_eval_state_data('speed', speeds[1:])
                self.add_to_eval_state_data('accel', accelerations[1:])
                self.add_to_eval_state_data('headway', headways)

                self.add_to_eval_leader('step', self.time_counter)
                self.add_to_eval_leader('accel', self.k.vehicle.get_realized_accel('leader_0'))
                self.add_to_eval_leader('speed', speeds[0])
                
                self.last_time = self.time_counter

        states = {
            self.veh_ids[1]: [accelerations[1], speeds[1], speeds[0], headways[0], speeds[0]],
            self.veh_ids[2]: [accelerations[2], speeds[2], speeds[1], headways[1], speeds[0]],
            self.veh_ids[3]: [accelerations[3], speeds[3], speeds[2], headways[2], speeds[0]],
            self.veh_ids[4]: [accelerations[4], speeds[4], speeds[3], headways[3], speeds[0]],
            self.veh_ids[5]: [accelerations[5], speeds[5], speeds[4], headways[4], speeds[0]]
        }

        return states
    


class PloegEnv(PlatoonEnv):

    def __init__(self, env_params, sim_params, network, simulator='traci'):
        super().__init__(env_params, sim_params, network, simulator)

        self.standstill_distance = 2
        self.time_gap = 1

    @property
    def observation_space(self):
        return Box(
            low=-1000000, # an arbitrary high enough number
            high=1000000, # an arbitrary high enough number
            shape=(5, ),
            dtype=np.float32)
    

    def compute_reward(self, rl_actions, **kwargs):

        # retrieve state information
        accelerations = [(0 if not isinstance(self.k.vehicle.get_arrived_ids(), int) and veh_id in self.k.vehicle.get_arrived_ids()
                         else self.k.vehicle.get_accel(veh_id, noise=False, failsafe=False))
                        for veh_id in self.veh_ids]
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])
        speeds = self.k.vehicle.get_speed(self.veh_ids)

        # in case of a collision
        headways = [(headway if headway >= 0 else 0) for headway in headways]

        # identify crashed vehicles
        crashed = {}
        for veh_id in self.veh_ids[1:]:
            crashed[veh_id] = veh_id in self.k.vehicle.get_arrived_rl_ids(self.env_params.sims_per_step)

        # compute rewards
        reward_follower0, gap_error0, speed_error0, jerk0 = reward_function_unilateral(headways[0], speeds[0], speeds[1], accelerations[1], self.previous_accels[self.veh_ids[1]], self.time_gap, self.standstill_distance)
        reward_follower1, gap_error1, speed_error1, jerk1 = reward_function_unilateral(headways[1], speeds[1], speeds[2], accelerations[2], self.previous_accels[self.veh_ids[2]], self.time_gap, self.standstill_distance)
        reward_follower2, gap_error2, speed_error2, jerk2 = reward_function_unilateral(headways[2], speeds[2], speeds[3], accelerations[3], self.previous_accels[self.veh_ids[3]], self.time_gap, self.standstill_distance)
        reward_follower3, gap_error3, speed_error3, jerk3 = reward_function_unilateral(headways[3], speeds[3], speeds[4], accelerations[4], self.previous_accels[self.veh_ids[4]], self.time_gap, self.standstill_distance)
        reward_follower4, gap_error4, speed_error4, jerk4 = reward_function_unilateral(headways[4], speeds[4], speeds[5], accelerations[5], self.previous_accels[self.veh_ids[5]], self.time_gap, self.standstill_distance)
            
        rewards = {self.veh_ids[1]: reward_follower0,
                   self.veh_ids[2]: reward_follower1,
                   self.veh_ids[3]: reward_follower2,
                   self.veh_ids[4]: reward_follower3,
                   self.veh_ids[5]: reward_follower4
                   }
        
        # log evaluation data
        if self.env_params.evaluate:
            rl_actions_list = []
            for veh_id in self.veh_ids[1:]:
                rl_actions_list.append(rl_actions[veh_id][0])
            
            self.add_to_eval_reward_data('step', [self.time_counter]*len(self.veh_ids[1:]))
            self.add_to_eval_reward_data('input', rl_actions_list)
            self.add_to_eval_reward_data('gap_error', [gap_error0, gap_error1, gap_error2, gap_error3, gap_error4])
            self.add_to_eval_reward_data('speed_error', [speed_error0, speed_error1, speed_error2, speed_error3, speed_error4])
            self.add_to_eval_reward_data('jerk', [jerk0, jerk1, jerk2, jerk3, jerk4])
            self.add_to_eval_reward_data('reward', [reward_follower0, reward_follower1, reward_follower2, reward_follower3, reward_follower4])

        return rewards
    

    def get_state(self, **kwargs):

        # retrieve state information
        speeds = self.k.vehicle.get_speed(self.veh_ids)
        previous_speeds = self.k.vehicle.get_previous_speed(self.veh_ids)
        accelerations = [(0 if not isinstance(self.k.vehicle.get_arrived_ids(), int) and veh_id in self.k.vehicle.get_arrived_ids()
                         else self.k.vehicle.get_accel(veh_id, noise=False, failsafe=False))
                        for veh_id in self.veh_ids]
        headways = self.k.vehicle.get_headway(self.veh_ids[1:])

        # in case of a collision
        headways = [(headway if headway >= 0 else 0) for headway in headways]
        speeds = [(speed if speed >= 0 else previous_speeds[i]) for i, speed in enumerate(speeds)]

        # accelerations are not updated at the start
        accelerations = [0 if accelerations[i] is None else accelerations[i] for i in range(len(self.veh_ids))]

        # log state
        if self.env_params.evaluate:
            if self.last_time == self.time_counter:
                self.last_time = -1
            else:
                self.add_to_eval_state_data('step', [self.time_counter]*len(self.veh_ids[1:]))
                self.add_to_eval_state_data('speed', speeds[1:])
                self.add_to_eval_state_data('accel', accelerations[1:])
                self.add_to_eval_state_data('headway', headways)

                self.add_to_eval_leader('step', self.time_counter)
                self.add_to_eval_leader('accel', self.k.vehicle.get_realized_accel('leader_0'))
                self.add_to_eval_leader('speed', speeds[0])
                
                self.last_time = self.time_counter

        states = {
            self.veh_ids[1]: [speeds[1], speeds[0], accelerations[1], self.k.vehicle.get_realized_accel('leader_0'), headways[0]],
            self.veh_ids[2]: [speeds[2], speeds[1], accelerations[2], accelerations[1], headways[1]],
            self.veh_ids[3]: [speeds[3], speeds[2], accelerations[3], accelerations[2], headways[2]],
            self.veh_ids[4]: [speeds[4], speeds[3], accelerations[4], accelerations[3], headways[3]],
            self.veh_ids[5]: [speeds[5], speeds[4], accelerations[5], accelerations[4], headways[4]]
        }


        return states
    


