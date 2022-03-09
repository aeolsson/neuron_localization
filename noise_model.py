import numpy as np

from waveform_model import Waveform_model
from firing_model import Firing_model
import params

class Noise_model:
    def __init__(self,
                 electrode_positions,
                 r_o,
                 r_i,
                 z_min_o,
                 z_min_i,
                 z_max,
                 density,
                 min_rate,
                 max_rate,
                 thermal_noise=None):
        
        self.electrode_positions = electrode_positions
        self.num_electrodes = np.shape(electrode_positions)[0]
        
        self.r_o = r_o
        self.r_i = r_i
        self.z_min_o = z_min_o
        self.z_min_i = z_min_i
        self.z_max = z_max
        
        self.density = density
        
        self.min_rate = min_rate
        self.max_rate = max_rate
        
        self.thermal_noise = thermal_noise
        
        self.assign_positions()
        self.assign_types()
        self.compute_waveforms()
        
    
    def assign_positions(self):
        neuron_positions = []
        
        # Volume of cylinder wall
        height = self.z_max - self.z_min_i
        V_outer = height * np.pi * (self.r_o ** 2)
        V_inner = height * np.pi * (self.r_i ** 2)
        V_wall = V_outer - V_inner
        
        # Volume of bottom
        V_bottom = (self.z_min_i - self.z_min_o) * np.pi * (self.r_o ** 2)
        
        # Neurons in wall
        N_wall = int(np.round(V_wall * self.density))
        for _ in range(N_wall):
            r = self.r_o * np.sqrt(np.random.uniform(self.r_i**2/self.r_o**2, 1.0))
            theta = 2 * np.pi * np.random.uniform(0.0, 1.0)
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.uniform(self.z_min_i, self.z_max)
            
            neuron_positions.append(np.array([x, y, z]))
        
        # Neurons in bottom
        N_bottom = int(np.round(V_bottom * self.density))
        for _ in range(N_bottom):
            r = self.r_o * np.sqrt(np.random.uniform(0, 1.0))
            theta = 2 * np.pi * np.random.uniform(0.0, 1.0)
            
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.uniform(self.z_min_o, self.z_min_i)
            
            neuron_positions.append(np.array([x, y, z]))
        
        self.num_noise_neurons = N_wall + N_bottom
        
        self.neuron_positions = np.array(neuron_positions)
        np.random.shuffle(self.neuron_positions)
    
    def assign_types(self):
        self.types = -np.ones((self.num_noise_neurons), dtype=np.int8)
        self.waveform_models = []
        for i, position in enumerate(self.neuron_positions):
            model_idx = i % 4
            path = params.model_paths[model_idx]
            self.types[i] = model_idx
            
            waveform_model = Waveform_model.from_mat(path,
                                                     x=position[0],
                                                     y=position[1],
                                                     z=position[2])
            self.waveform_models.append(waveform_model)
    
    
    def compute_waveforms(self):
        waveforms = np.zeros((self.num_electrodes, self.num_noise_neurons, params.waveform_duration))
        for electrode_idx in range(self.num_electrodes):
            x_e = self.electrode_positions[electrode_idx, 0]
            y_e = self.electrode_positions[electrode_idx, 1]
            z_e = self.electrode_positions[electrode_idx, 2]
            for neuron_idx in range(self.num_noise_neurons):
                waveform = self.waveform_models[neuron_idx].get_waveform(x_e, y_e, z_e)
                waveforms[electrode_idx, neuron_idx, :] = waveform
                
        self.waveforms = waveforms
    
    def sample(self, duration):
        spike_trains = self.generate_spike_trains(duration=duration)
        
        signal_matrix = np.zeros((self.num_electrodes, duration * params.fs))
        
        if self.thermal_noise:
            signal_matrix += np.random.normal(loc=0, scale=self.thermal_noise, size=tuple(signal_matrix.shape))
        
        for electrode_idx in range(self.num_electrodes):
            for neuron_idx in range(self.num_noise_neurons):
                event_waveform = self.waveforms[electrode_idx, neuron_idx, :]
                spike_train = spike_trains[neuron_idx, :]
                
                signal_matrix[electrode_idx, :] += np.convolve(spike_train, event_waveform, mode='same')
        
        return signal_matrix, spike_trains
    
    def generate_spike_trains(self, duration):
        fm = Firing_model(firing_rate=[self.min_rate, self.max_rate], shape_parameter=5)

        spike_trains = np.zeros((self.num_noise_neurons, int(params.fs*duration)))
        for neuron_idx in range(self.num_noise_neurons):
            spike_trains[neuron_idx, :] = fm.generate_spike_train(duration=duration)
        
        return spike_trains
            