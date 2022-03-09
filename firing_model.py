import numpy as np
import params

class Firing_model:
    
    def __init__(self,
                 firing_rate=[1],
                 shape_parameter=5):
        self.firing_rate = firing_rate
        self.shape_parameter = shape_parameter
        
    def generate_spike_train(self, duration):
        spike_times = self.generate_spike_times(duration)
        spike_train = np.zeros((int(params.fs * duration)))
        spike_train[[min(int(params.fs * spike_time), int(params.fs*duration)-1) for spike_time in spike_times]] = 1.0
        return spike_train
    
    def generate_spike_times(self, duration):
        
        ISIs = self.generate_ISIs(duration)
        
        spike_times = np.cumsum(ISIs)
        spike_times = spike_times[spike_times < duration]
        
        return spike_times
    
    def generate_ISIs(self, duration):
        if len(self.firing_rate) == 1:
            firing_rate = self.firing_rate[0]
        else:
            firing_rate = np.random.uniform(self.firing_rate[0], self.firing_rate[1])
        
        k = self.shape_parameter
        ISI_mean = 1 / firing_rate
        theta = ISI_mean / k
        
        # Estimated number of spikes
        N = int(duration / ISI_mean) + 5
        
        ISIs = np.random.gamma(shape=k,
                              scale=theta,
                              size=N)
        return ISIs
        
        
        