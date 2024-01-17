import numpy as np
from abc import ABC, abstractmethod 


class BaseScenario(ABC):

    @abstractmethod
    def get_accel(self, step, speed):
        pass


class ConstantSpeedScenario(BaseScenario):

    def __init__(self):
        self.name = 'ConstantSpeedScenario'

    def get_accel(self, step, speed):
        return 0
    

class BrakingScenario(BaseScenario):

    def __init__(self):
        self.name = 'BrakingScenario'
        
        self.start_time = np.random.randint(low=50, high=250)
        self.upper_deceleration = 3
        self.lower_deceleration = 1
        self.lower_speed_bound = np.random.uniform(low=0, high=10)

    def get_accel(self, step, speed):
        
        if step < self.start_time or speed < self.lower_speed_bound:
            return 0
        
        return -np.random.uniform(low=self.lower_deceleration, high=self.upper_deceleration)
  

class AccelerationScenario(BaseScenario):
    def __init__(self):
        self.name = 'AccelerationScenario'
        
        self.start_time = np.random.randint(low=50, high=250)
        self.upper_acceleration = 3
        self.lower_acceleration = 1
        self.upper_speed_bound = np.random.uniform(low=22, high=33)

    def get_accel(self, step, speed):
        
        if step < self.start_time or speed > self.upper_speed_bound:
            return 0
        
        return np.random.uniform(low=self.lower_acceleration, high=self.upper_acceleration)
    

class AccelerationAndBrakingScenario(BaseScenario):
    def __init__(self):
        self.name = 'AccelerationAndBrakingScenario'
        
        self.start_time = np.random.randint(low=50, high=100)
        self.acceleration_time = np.random.randint(low=200, high=300)
        self.idle_time = np.random.randint(low=0, high=100)
        self.upper_acceleration = 3
        self.lower_acceleration = 1
        self.upper_deceleration = 3
        self.lower_deceleration = 1
        self.upper_speed_bound = np.random.uniform(low=22, high=33)
        self.lower_speed_bound = np.random.uniform(low=0, high=10)

    def get_accel(self, step, speed):

        if step < self.start_time:
            return 0
        if step < self.start_time + self.acceleration_time:
            if speed > self.upper_speed_bound:
                return 0
            return np.random.uniform(low=self.lower_acceleration, high=self.upper_acceleration)
        if step < self.start_time + self.acceleration_time + self.idle_time:
            return 0
        if speed < self.lower_speed_bound:
            return 0
        
        return -np.random.uniform(low=self.lower_deceleration, high=self.upper_deceleration)


class BrakingAndAccelerationScenario(BaseScenario):
    def __init__(self):
        self.name = 'BrakingAndAccelerationScenario'
        
        self.start_time = np.random.randint(low=50, high=100)
        self.braking_time = np.random.randint(low=200, high=300)
        self.idle_time = np.random.randint(low=0, high=100)
        self.upper_acceleration = 3
        self.lower_acceleration = 1
        self.upper_deceleration = 3
        self.lower_deceleration = 1
        self.upper_speed_bound = np.random.uniform(low=22, high=33)
        self.lower_speed_bound = np.random.uniform(low=0, high=10)


    def get_accel(self, step, speed):

        if step < self.start_time:
            return 0
        if step < self.start_time + self.braking_time:
            if speed < self.lower_speed_bound:
                return 0
            return -np.random.uniform(low=self.lower_deceleration, high=self.upper_deceleration)
        if step < self.start_time + self.braking_time + self.idle_time:
            return 0
        if speed > self.upper_speed_bound:
            return 0
        
        return np.random.uniform(low=self.lower_acceleration, high=self.upper_acceleration)

    
class SinusoidalScenario(BaseScenario):
    def __init__(self):
        self.name = 'SinusoidalScenario'
        
        self.start_time = np.random.randint(low=50, high=100)
        self.amplitude = np.random.uniform(low=1, high=3)
        self.period = np.random.randint(low=100, high=200)
        self.sinus = np.random.choice(a=[True, False])


    def get_accel(self, step, speed):
        if step < self.start_time:
            return 0

        if self.sinus:
            return np.sin((2 * np.pi) * ((step - self.start_time) / self.period)) * self.amplitude
        else:
            return np.cos((2 * np.pi) * ((step - self.start_time) / self.period)) * self.amplitude



