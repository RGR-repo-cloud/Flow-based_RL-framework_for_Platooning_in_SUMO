import numpy as np
from abc import ABC, abstractmethod 


class BaseScenario(ABC):

    @abstractmethod
    def get_accel(self, step, speed):
        pass


class RandomizedSpeedScenario(BaseScenario):

    def __init__(self):
        self.name = 'RandomizedSpeedScenario'
        
        self.accel_variance = 6

    def get_accel(self, step, speed):
        return np.random.uniform(low=-self.accel_variance/2, high=self.accel_variance/2)


class StaticSpeedScenario(BaseScenario):

    def __init__(self):
        self.name = 'StaticSpeedScenario'

    def get_accel(self, step, speed):
        return 0
    

class RandomizedBrakingScenario(BaseScenario):

    def __init__(self):
        self.name = 'RandomizedBrakingScenario'
        
        self.start_time = np.random.randint(low=50, high=250)
        self.upper_deceleration = 3
        self.lower_deceleration = 2
        self.lower_speed_bound = np.random.uniform(low=0, high=4)

    def get_accel(self, step, speed):
        
        if step < self.start_time or speed < self.lower_speed_bound:
            return 0
        
        return -np.random.uniform(low=self.lower_deceleration, high=self.upper_deceleration)
    

class StaticBrakingScenario(BaseScenario):

    def __init__(self):
        self.name = 'StaticBrakingScenario'
        
        self.start_time = 120
        self.lower_speed_bound = 2
        self.deceleration = 3

    def get_accel(self, step, speed):
        
        if step < self.start_time or speed < self.lower_speed_bound:
            return 0
        
        return -self.deceleration
  

class RandomizedAccelerationScenario(BaseScenario):
    def __init__(self):
        self.name = 'RandomizedAccelerationScenario'
        
        self.start_time = np.random.randint(low=50, high=250)
        self.upper_acceleration = 3
        self.lower_acceleration = 2
        self.upper_speed_bound = np.random.uniform(low=25, high=33)

    def get_accel(self, step, speed):
        
        if step < self.start_time or speed > self.upper_speed_bound:
            return 0
        
        return np.random.uniform(low=self.lower_acceleration, high=self.upper_acceleration)
    

class StaticAccelerationScenario(BaseScenario):
    def __init__(self):
        self.name = 'StaticAccelerationScenario'
        
        self.start_time = 120
        self.upper_speed_bound = 33
        self.acceleration = 3

    def get_accel(self, step, speed):
        
        if step < self.start_time or speed > self.upper_speed_bound:
            return 0
        
        return self.acceleration
    

class RandomizedAccelerationAndBrakingScenario(BaseScenario):
    def __init__(self):
        self.name = 'RandomizedAccelerationAndBrakingScenario'
        
        self.start_time = np.random.randint(low=50, high=100)
        self.acceleration_time = np.random.randint(low=200, high=300)
        self.idle_time = np.random.randint(low=0, high=100)
        self.upper_acceleration = 3
        self.lower_acceleration = 1
        self.upper_deceleration = 3
        self.lower_deceleration = 1
        self.upper_speed_bound = np.random.uniform(low=25, high=33)
        self.lower_speed_bound = np.random.uniform(low=0, high=4)

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
    

class StaticAccelerationAndBrakingScenario(BaseScenario):
    def __init__(self):
        self.name = 'StaticAccelerationAndBrakingScenario'
        
        self.start_time = 50
        self.acceleration_time = 250
        self.idle_time = 50
        self.acceleration = 3
        self.deceleration = 3
        self.upper_speed_bound = 28
        self.lower_speed_bound = 2

    def get_accel(self, step, speed):

        if step < self.start_time:
            return 0
        if step < self.start_time + self.acceleration_time:
            if speed > self.upper_speed_bound:
                return 0
            return self.acceleration
        if step < self.start_time + self.acceleration_time + self.idle_time:
            return 0
        if speed < self.lower_speed_bound:
            return 0
        
        return -self.deceleration


class RandomizedBrakingAndAccelerationScenario(BaseScenario):
    def __init__(self):
        self.name = 'BrakingAndAccelerationScenario'
        
        self.start_time = np.random.randint(low=50, high=100)
        self.braking_time = np.random.randint(low=200, high=300)
        self.idle_time = np.random.randint(low=0, high=100)
        self.upper_acceleration = 3
        self.lower_acceleration = 1
        self.upper_deceleration = 3
        self.lower_deceleration = 1
        self.upper_speed_bound = np.random.uniform(low=25, high=33)
        self.lower_speed_bound = np.random.uniform(low=0, high=4)


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


class StaticBrakingAndAccelerationScenario(BaseScenario):
    def __init__(self):
        self.name = 'StaticBrakingAndAccelerationScenario'
        
        self.start_time = 50
        self.braking_time = 250
        self.idle_time = 50
        self.acceleration = 3
        self.deceleration = 3
        self.upper_speed_bound = 28
        self.lower_speed_bound = 2


    def get_accel(self, step, speed):

        if step < self.start_time:
            return 0
        if step < self.start_time + self.braking_time:
            if speed < self.lower_speed_bound:
                return 0
            return -self.deceleration
        if step < self.start_time + self.braking_time + self.idle_time:
            return 0
        if speed > self.upper_speed_bound:
            return 0
        
        return self.acceleration

    
class RandomizedSinusoidalScenario(BaseScenario):
    def __init__(self):
        self.name = 'RandomizedSinusoidalScenario'
        
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
        

class StaticSinusoidalScenario(BaseScenario):
    def __init__(self):
        self.name = 'StaticSinusoidalScenario'
        
        self.start_time = 50
        self.amplitude = 3
        self.period = 100


    def get_accel(self, step, speed):
        if step < self.start_time:
            return 0

        return np.sin((2 * np.pi) * ((step - self.start_time) / self.period)) * self.amplitude



