import numpy as np
from abc import ABC, abstractmethod 


class RandomizedScenario(ABC):

    def __init__(self, randomizer):
        self.randomizer = randomizer

    @abstractmethod
    def get_accel(self, step, speed):
        pass

class StaticScenario(ABC):

    @abstractmethod
    def get_accel(self, step, speed):
        pass


class RandomizedSpeedScenario(RandomizedScenario):

    def __init__(self, randomizer):
        super().__init__(randomizer)
        self.name = 'RandomizedSpeedScenario'
        
        self.accel_variance = 6

    def get_accel(self, step, speed):
        return self.randomizer.uniform(low=-self.accel_variance/2, high=self.accel_variance/2)


class StaticSpeedScenario(StaticScenario):

    def __init__(self):
        self.name = 'StaticSpeedScenario'

    def get_accel(self, step, speed):
        return 0
    

class RandomizedBrakingScenario(RandomizedScenario):

    def __init__(self, randomizer):
        super().__init__(randomizer)
        self.name = 'RandomizedBrakingScenario'
        
        self.start_time = self.randomizer.integers(low=50, high=250)
        self.upper_deceleration = 3
        self.lower_deceleration = 2
        self.lower_speed_bound = self.randomizer.uniform(low=0, high=4)

    def get_accel(self, step, speed):
        
        if step < self.start_time or speed < self.lower_speed_bound:
            return 0
        
        return -self.randomizer.uniform(low=self.lower_deceleration, high=self.upper_deceleration)
    

class StaticBrakingScenario(StaticScenario):

    def __init__(self):
        self.name = 'StaticBrakingScenario'
        
        self.start_time = 120
        self.lower_speed_bound = 2
        self.deceleration = 3

    def get_accel(self, step, speed):
        
        if step < self.start_time or speed < self.lower_speed_bound:
            return 0
        
        return -self.deceleration
  

class RandomizedAccelerationScenario(RandomizedScenario):
    def __init__(self, randomizer):
        super().__init__(randomizer)
        self.name = 'RandomizedAccelerationScenario'
        
        self.start_time = self.randomizer.integers(low=50, high=250)
        self.upper_acceleration = 3
        self.lower_acceleration = 2
        self.upper_speed_bound = self.randomizer.uniform(low=25, high=33)

    def get_accel(self, step, speed):
        
        if step < self.start_time or speed > self.upper_speed_bound:
            return 0
        
        return self.randomizer.uniform(low=self.lower_acceleration, high=self.upper_acceleration)
    

class StaticAccelerationScenario(StaticScenario):
    def __init__(self):
        self.name = 'StaticAccelerationScenario'
        
        self.start_time = 120
        self.upper_speed_bound = 33
        self.acceleration = 3

    def get_accel(self, step, speed):
        
        if step < self.start_time or speed > self.upper_speed_bound:
            return 0
        
        return self.acceleration
    

class RandomizedAccelerationAndBrakingScenario(RandomizedScenario):
    def __init__(self, randomizer):
        super().__init__(randomizer)
        self.name = 'RandomizedAccelerationAndBrakingScenario'
        
        self.start_time = self.randomizer.integers(low=50, high=100)
        self.acceleration_time = self.randomizer.integers(low=200, high=300)
        self.idle_time = self.randomizer.integers(low=0, high=100)
        self.upper_acceleration = 3
        self.lower_acceleration = 1
        self.upper_deceleration = 3
        self.lower_deceleration = 1
        self.upper_speed_bound = self.randomizer.uniform(low=25, high=33)
        self.lower_speed_bound = self.randomizer.uniform(low=0, high=4)

    def get_accel(self, step, speed):

        if step < self.start_time:
            return 0
        if step < self.start_time + self.acceleration_time:
            if speed > self.upper_speed_bound:
                return 0
            return self.randomizer.uniform(low=self.lower_acceleration, high=self.upper_acceleration)
        if step < self.start_time + self.acceleration_time + self.idle_time:
            return 0
        if speed < self.lower_speed_bound:
            return 0
        
        return -self.randomizer.uniform(low=self.lower_deceleration, high=self.upper_deceleration)
    

class StaticAccelerationAndBrakingScenario(StaticScenario):
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


class RandomizedBrakingAndAccelerationScenario(RandomizedScenario):
    def __init__(self, randomizer):
        super().__init__(randomizer)
        self.name = 'RandomizedBrakingAndAccelerationScenario'
        
        self.start_time = self.randomizer.integers(low=50, high=100)
        self.braking_time = self.randomizer.integers(low=200, high=300)
        self.idle_time = self.randomizer.integers(low=0, high=100)
        self.upper_acceleration = 3
        self.lower_acceleration = 1
        self.upper_deceleration = 3
        self.lower_deceleration = 1
        self.upper_speed_bound = self.randomizer.uniform(low=25, high=33)
        self.lower_speed_bound = self.randomizer.uniform(low=0, high=4)


    def get_accel(self, step, speed):

        if step < self.start_time:
            return 0
        if step < self.start_time + self.braking_time:
            if speed < self.lower_speed_bound:
                return 0
            return -self.randomizer.uniform(low=self.lower_deceleration, high=self.upper_deceleration)
        if step < self.start_time + self.braking_time + self.idle_time:
            return 0
        if speed > self.upper_speed_bound:
            return 0
        
        return self.randomizer.uniform(low=self.lower_acceleration, high=self.upper_acceleration)


class StaticBrakingAndAccelerationScenario(StaticScenario):
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

    
class RandomizedSinusoidalScenario(RandomizedScenario):
    def __init__(self, randomizer):
        super().__init__(randomizer)
        self.name = 'RandomizedSinusoidalScenario'
        
        self.start_time = self.randomizer.integers(low=50, high=100)
        self.amplitude = self.randomizer.uniform(low=1, high=3)
        self.period = self.randomizer.integers(low=100, high=200)
        self.sinus = self.randomizer.choice(a=[True, False])


    def get_accel(self, step, speed):
        if step < self.start_time:
            return 0

        if self.sinus:
            return np.sin((2 * np.pi) * ((step - self.start_time) / self.period)) * self.amplitude
        else:
            return np.cos((2 * np.pi) * ((step - self.start_time) / self.period)) * self.amplitude
        

class StaticSinusoidalScenario(StaticScenario):
    def __init__(self):
        self.name = 'StaticSinusoidalScenario'
        
        self.start_time = 50
        self.amplitude = 3
        self.period = 100
        #self.period = 200 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    def get_accel(self, step, speed):
        if step < self.start_time:
            return 0

        return np.sin((2 * np.pi) * ((step - self.start_time) / self.period)) * self.amplitude



