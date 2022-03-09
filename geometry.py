import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 

from waveform_model import Waveform_model
import params

class Geometry:
    def __init__(self, electrode_positions, neuron_positions, neuron_types, firing_models):
        self.firing_models = firing_models
        
        self.electrode_positions = electrode_positions
        self.neuron_positions = neuron_positions
        
        self.num_electrodes = np.shape(electrode_positions)[0]
        self.num_neurons = np.shape(neuron_positions)[0]
        
        xs = neuron_positions[:, 0]
        ys = neuron_positions[:, 1]
        zs = neuron_positions[:, 2]
        self.waveform_models = [Waveform_model.from_mat(params.model_paths[neuron_types[i]], xs[i], ys[i], zs[i]) for i in range(self.num_neurons)]
        
        self.compute_waveforms()
    
    def compute_waveforms(self):
        waveforms = np.zeros((self.num_electrodes, self.num_neurons, params.waveform_duration))
        for electrode_idx in range(self.num_electrodes):
            x_e = self.electrode_positions[electrode_idx, 0]
            y_e = self.electrode_positions[electrode_idx, 1]
            z_e = self.electrode_positions[electrode_idx, 2]
            for neuron_idx in range(self.num_neurons):
                waveform = self.waveform_models[neuron_idx].get_waveform(x_e, y_e, z_e)
                waveforms[electrode_idx, neuron_idx, :] = waveform
                
        self.waveforms = waveforms
                
    
    def sample(self, duration=60):
        spike_trains = self.generate_spike_trains(duration=duration)
        
        signal_matrix = np.zeros([self.num_electrodes, int(params.fs*duration)])
        
        for electrode_idx in range(self.num_electrodes): 
            for neuron_idx in range(self.num_neurons):
                event_waveform = self.waveforms[electrode_idx, neuron_idx, :]
#                event_waveform = np.concatenate([np.zeros(params.waveform_duration), event_waveform])
                spike_train = spike_trains[neuron_idx, :]
                
                signal_matrix[electrode_idx, :] += np.convolve(spike_train, event_waveform, mode='same')
        
        return signal_matrix, spike_trains
    
    def generate_spike_trains(self, duration):
        spike_trains = np.zeros((self.num_neurons, int(params.fs*duration)))
        
        for neuron_idx in range(self.num_neurons):
            fm = self.firing_models[neuron_idx]
            spike_trains[neuron_idx, :] = fm.generate_spike_train(duration=duration)
        
        return spike_trains
    
    def plot(self, ax=None):
        if not ax:
            ax = plt.axes(projection='3d')
        
        es = self.electrode_positions
        ns = self.neuron_positions
        
        ax.scatter(xs=es[:, 0],
                   ys=es[:, 1],
                   zs=es[:, 2],
                   marker='o',
                   color='blue',
                   s=200,
                   depthshade=False,
                   label='Electrodes')
        
        ax.scatter(xs=ns[:, 0],
                   ys=ns[:, 1],
                   zs=ns[:, 2],
                   marker='X',
                   color='red',
                   s=400,
                   depthshade=False,
                   label='Neurons')
        for n in range(np.shape(ns)[0]):
            ax.plot(xs=[ns[n, 0], ns[n, 0]], ys=[ns[n, 1], ns[n, 1]], zs=[ns[n, 2], -60], color='black', linewidth=1.5)
        
        for e in range(np.shape(es)[0]):
            ax.plot(xs=[es[e, 0], es[e, 0]], ys=[es[e, 1], es[e, 1]], zs=[es[e, 2], -60], color='black', linewidth=1.5)
        
        return ax