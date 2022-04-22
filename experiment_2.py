import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
import numpy as np
#from scipy.signal import butter, filtfilt
from scipy.optimize import fmin_cg, dual_annealing
import time

import params
import utils

from waveform_model import Waveform_model
from firing_model import Firing_model

from geometry import Geometry
from noise_model import Noise_model

from spike_detectors import Multichannel_spike_detector
from spike_sorters import PCA_spike_sorter

duration = 60 #s

for _ in range(1):
    t_i = time.time()
    # Generate electrode locations
    num_electrodes = 8
    hs = np.linspace(-50, 50, num_electrodes)
    true_electrode_positions = np.array([(0, 0, h) for h in hs])
    
    # Generate neuron properties
    r_n = 25
    num_neurons = 3
#    xi1s = np.random.uniform(low=0.0, high=1.0, size=num_neurons)
#    xi2s = np.random.uniform(low=0.0, high=1.0, size=num_neurons)
#    xi3s = np.random.uniform(low=0.0, high=1.0, size=num_neurons)
#    rs = [r_n*np.sqrt(xi1) for xi1 in xi1s]
#    thetas = [np.pi*xi2 for xi2 in xi2s]
#    hs = [-50 + (100*xi3) for xi3 in xi3s]
#    true_neuron_positions = np.array([[r*np.cos(theta), r*np.sin(theta), h] for r, theta, h in zip(rs, thetas, hs)])
    
    true_neuron_positions = r_n * np.array([[-1/np.sqrt(2), 1/np.sqrt(2), -1],
                                            [1, 0, 0],
                                            [0, -1, 1]])
    
    neuron_types = num_neurons*[0]
    
    firing_models = [Firing_model(firing_rate=[10], shape_parameter=5) for _ in range(num_neurons)]
    
    # Construct measurement setup geometry
    true_geometry = Geometry(electrode_positions=true_electrode_positions,
                             neuron_positions=true_neuron_positions,
                             neuron_types=neuron_types,
                             firing_models=firing_models)
    
    # Set up population of noise neurons
    noise_model = Noise_model(electrode_positions=true_electrode_positions,
                              r_o=250,
                              r_i=120,
                              z_min_o=-250,
                              z_min_i=-250,
                              z_max=250,
                              density=9.5e-6,
                              min_rate=1,
                              max_rate=50,
                              thermal_noise=1e-3)
    
    t_f = time.time()
    print('Time to set up geometry: {:.2f}'.format(t_f - t_i))
    t_i = time.time()
    
    # Sample combined system (geometry + noise)
    signal, real_spike_trains = true_geometry.sample(duration=duration)
    noise, _ = noise_model.sample(duration=duration)
    measurement = signal + noise
    measurement = (measurement.T - np.mean(measurement.T, axis=0)).T
    
    num_samples = measurement.shape[1]
    
    t_f = time.time()
    print('Time to generate measurement: {:.2f}'.format(t_f - t_i))
    t_i = time.time()
    
    # Estimate firing times from measurement
    detector = Multichannel_spike_detector(polarity=-1, threshold=-10)
    est_spike_times, thresholds = detector.detect_spike_times(measurement)
    waveforms = detector.get_waveforms(measurement, est_spike_times)
    
    t_f = time.time()
    print('Time to detect spikes: {:.2f}'.format(t_f - t_i))
    t_i = time.time()
    
    # Sort spikes according to neuron
    spike_sorter = PCA_spike_sorter(num_sources=num_neurons)
    membership, components = spike_sorter.sort(waveforms)
    
    
    t_f = time.time()
    print('Time to sort spikes: {:.2f}'.format(t_f - t_i))
    t_i = time.time()
    
    
    archetypes = []
    est_spike_trains = []
    for c in range(num_neurons):
        rel_ind = np.where(membership==c)[0]
        
        relevant_waveforms = list(np.array(waveforms)[rel_ind, ...])
        archetype = sum(relevant_waveforms) / len(relevant_waveforms)
        archetypes.append(archetype)
        
        relevant_spike_times = list(np.array(est_spike_times)[rel_ind])
        relevant_spike_train = utils.times2spikes(relevant_spike_times, num_samples)
        est_spike_trains.append(relevant_spike_train)
    
    est_spike_trains = np.array(est_spike_trains)
    
    t_f = time.time()
    print('Time to estimate waveform morphologies: {:.2f}'.format(t_f - t_i))
    t_i = time.time()
    
    est_neuron_locations = []
    for i, archetype in enumerate(archetypes):
        def f(guessed_position):
            x_n, y_n, z_n = guessed_position
            
            path = params.model_paths[0]
            
            guessed_model = Waveform_model.from_mat(path=path,
                                                    x=x_n,
                                                    y=y_n,
                                                    z=z_n)
            
            cost = 0
            for e in range(num_electrodes):
                target_wf = archetype[e, :]#true_geometry.waveforms[e, 0, :]
                target_wf = target_wf - np.mean(target_wf)
                
                
                guessed_wf = guessed_model.get_waveform(x=true_electrode_positions[e, 0],
                                                        y=true_electrode_positions[e, 1],
                                                        z=true_electrode_positions[e, 2])
                guessed_wf = guessed_wf - np.mean(guessed_wf)
                
                c = np.square(target_wf - guessed_wf) / (np.std(target_wf)*np.std(guessed_wf))
                c = np.mean(c)
                cost += c
            
            cost = cost / num_electrodes
            return cost
        
        x0 = 0.95*true_electrode_positions[np.random.choice(num_electrodes), :]
        xf, allvecs = fmin_cg(f=f,
                              x0=x0,
                              gtol=1e-9,
                              maxiter=1000000,
                              retall=True)
        est_neuron_locations.append(xf)
        
        t_f = time.time()
        print('Time to locate source {}: {:.2f}'.format(i, t_f - t_i))
        t_i = time.time()
    
    print('True neuron positions {}'.format(true_neuron_positions))
    print('Estimated neuron positions {}'.format(est_neuron_locations))
    
    # Plot geometry
    ax = true_geometry.plot()
    
    ax.axes.set_xlim3d(left=-56, right=56) 
    ax.axes.set_ylim3d(bottom=-56, top=56) 
    ax.axes.set_zlim3d(bottom=-56, top=56) 
    
    ax.set_aspect('equal')
    
    ax.set_xlabel('X (um)', fontsize=24)
    ax.set_ylabel('Y (um)', fontsize=24)
    ax.set_zlabel('Z (um)', fontsize=24)
    ax.tick_params(axis='both', which='both', labelsize=15)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), fontsize=24, framealpha=1.0)
    
    # Plot simulated measurement
    num_plot_signals = 5
    selected_channels = np.random.choice(a=num_electrodes, size=num_plot_signals, replace=False)
    x = np.arange(0, duration, 1 / params.fs)
    F = plt.figure()
    for i, e in enumerate(selected_channels):
        ax = F.add_subplot(num_plot_signals, 1, i+1)
        
        ax.axis([0.0, 0.5, np.min(measurement), np.max(measurement)])
        ax.set_ylabel('Electrode {}'.format(e+1), fontsize=14)
        
        threshold = thresholds[e]
        
        y = measurement[e, :]
        if i == 0:
            ax.plot(x, y, color='blue', linewidth=1.0, label='Measured voltage', zorder=3)
            ax.axhline(y=threshold, xmin=0, xmax=params.fs*duration, color='black', linewidth=1.0, label='Threshold', zorder=1)
            for i, st in enumerate(list(est_spike_times)):
                if i==0:
                    ax.axvline(x=st/params.fs, ymin=0, ymax=1, color='red', linewidth=1.0, label='Detected spike', zorder=1)
                else:
                    ax.axvline(x=st/params.fs, ymin=0, ymax=1, color='red', linewidth=1.0, zorder=1)
            
            ax.legend(loc='upper center',
                      fontsize=24,
                      bbox_to_anchor=(0.5, 1.65),
                      ncol=3,
                      framealpha=1.0)
        else:
            ax.plot(x, y, color='blue', linewidth=1.0, zorder=3)
            ax.axhline(y=threshold, xmin=0, xmax=params.fs*duration, color='black', linewidth=1.0, zorder=1)
            for st in list(est_spike_times):
                ax.axvline(x=st/params.fs, ymin=0, ymax=1, color='red', linewidth=1.0, zorder=1)
        
        
        
        ax.set_yticks([])
        
        if i == num_plot_signals-1:
            ax.set_xlabel('Time (s)', fontsize=24)
        else:
            ax.set_xticks([])
    
    # Plot clusters
    colors = ['red', 'blue', 'green']
    F = plt.figure()
    ax = F.add_subplot(1, 1, 1)
    cl1 = np.where(membership==0)[0]
    cl2 = np.where(membership==1)[0]
    cl3 = np.where(membership==2)[0]
    
    ax.scatter(x=components[cl1, 0], y=components[cl1, 1], c=colors[0], edgecolors= "black", label='Cluster 1')
    ax.scatter(x=components[cl2, 0], y=components[cl2, 1], c=colors[1], edgecolors= "black", label='Cluster 2')
    ax.scatter(x=components[cl3, 0], y=components[cl3, 1], c=colors[2], edgecolors= "black", label='Cluster 3')
    
    ax.set_xlabel('PC1', fontsize=24)
    ax.set_ylabel('PC2', fontsize=24)
    
    ax.legend(loc='upper right', fontsize=24, framealpha=1.0)
    ax.tick_params(axis='both', which='both', labelsize=18)
        
#    wm = plt.get_current_fig_manager()
#    wm.window.state('zoomed')
#    plt.pause(1.0)
#    plt.tight_layout()
    
    # Compare real and detected sources
    F = plt.figure()
    x = np.arange(0, duration, 1 / params.fs)
    for n in range(num_neurons):
        ax = F.add_subplot(num_neurons, 2, 1 + 2*n)
        y = real_spike_trains[n, :]
        ax.plot(x, y, color='red', linewidth=3.0)
        ax.axis([0.0, 1.0, 0.0, 1.2])
        ax.set_title('Neuron {}'.format(n+1), fontsize=24)
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax = F.add_subplot(num_neurons, 2, 2+2*n)
        y = est_spike_trains[n, :]
        ax.plot(x, y, color='black', linewidth=3.0)
        ax.axis([0.0, 1.0, 0.0, 1.2])
        ax.set_title('Detected unit {}'.format(n+1), fontsize=24)
        ax.set_xticks([])
        ax.set_yticks([])
        
    # Compare real and estimated waveforms
    num_plot_signals = 5
    selected_channels = np.random.choice(a=num_electrodes, size=num_plot_signals, replace=False)
    F = plt.figure()
    index_map = [1, 0, 2]
    for i, e in enumerate(selected_channels):
        for j, n in enumerate(range(num_neurons)):
            est_index = index_map[n]
            ax = F.add_subplot(num_plot_signals, num_neurons, i*num_neurons + j + 1)
            ax.axis([0, 100, np.min(true_geometry.waveforms), np.max(true_geometry.waveforms)])
            
            if j == 0:
                ax.set_ylabel('Electrode {}'.format(e+1), fontsize=15)
            if i == 0:
                ax.set_title('Neuron {}'.format(n+1), fontsize=32)
            
            y_true = true_geometry.waveforms[e, n, :]
            y_est = archetypes[est_index][e, :]
            
            if i==0 and j==num_neurons-1:
                ax.plot(y_true, color='red', linewidth=2.0, zorder=1, label='True waveform')
                ax.plot(y_est, color='blue', linewidth=2.0, zorder=2, label='Estimated waveform')
                
#                ax.legend(loc='upper right', fontsize=20, framealpha=1.0)
            else:
                ax.plot(y_true, color='red', linewidth=2.0, zorder=1)
                ax.plot(y_est, color='blue', linewidth=2.0, zorder=2)
            

            ax.set_xticks([])
            ax.set_yticks([])
    
    
    t_f = time.time()
    print('Time to plot: {:.2f}'.format(t_f - t_i))
    break