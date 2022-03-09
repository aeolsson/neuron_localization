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
    samples_circle = 5
    samples_height = 3
    num_electrodes = samples_circle * samples_height
    r_e = 50
    
    thetas = np.linspace(0, 2*np.pi, samples_circle+1)[:-1]
    heights = np.linspace(-50, 50, samples_height)
    
    true_electrode_positions = []
    for theta in thetas:
        for h in heights:
            position = np.array((r_e*np.cos(theta), r_e*np.sin(theta), h))
            true_electrode_positions.append(position)
    true_electrode_positions = np.array(true_electrode_positions)
    
    # Generate neuron properties
    num_neurons = 1
    
    true_neuron_positions = np.array([[0, 0, 0],])
    
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
    
    
    archetype = sum(waveforms) / len(waveforms)
    
    noise_levels = np.linspace(0, 0.002, 11)
    num_perturbations = 10
    archetypes_perturbed = []
    for sigma in noise_levels:
        archetypes_at_level = []
        for _ in range(num_perturbations):
            archetype_perturbed = archetype.copy()
            for e in range(num_electrodes):
                waveform = archetype_perturbed[e, :]
                
                white_noise = np.random.normal(loc=0, scale=sigma, size=waveform.size)
                brownian_noise = np.cumsum(white_noise)
                
                perturbation = brownian_noise
                
                waveform_perturbed = waveform + perturbation
                archetype_perturbed[e, :] = waveform_perturbed
            
            archetypes_at_level.append(archetype_perturbed)
        archetypes_perturbed.append(archetypes_at_level)
    
    t_f = time.time()
    print('Time to estimate waveform morphologies: {:.2f}'.format(t_f - t_i))
    t_i = time.time()
    
    # Compare real and estimated waveforms
    num_plot_signals = 5
    selected_channels = np.random.choice(a=num_electrodes, size=num_plot_signals, replace=False)
    selected_channels.sort()
    F = plt.figure()
    for i, e in enumerate(selected_channels):
        for j, p in enumerate(np.arange(0, 11, 5)):
            ax = F.add_subplot(num_plot_signals, 3, i*3 + j + 1)
            ax.axis([0, 100, np.min(true_geometry.waveforms), np.max(true_geometry.waveforms)])
            
            if j == 0:
                ax.set_ylabel('Electrode {}'.format(e+1), fontsize=15)
            if i == 0:
                ax.set_title('Perturbation {}'.format(p+1), fontsize=32)
            
            y_true = true_geometry.waveforms[e, 0, :]
            y_est = archetypes_perturbed[p][0][e, :]
            
            if i==num_plot_signals-1 and j==1:
                ax.plot(y_true, color='red', linewidth=2.5, zorder=1, label='True spike waveform')
                ax.plot(y_est, color='blue', linewidth=2.5, linestyle='--', zorder=2, label='Estimated spike waveform')
                
                ax.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.75), fontsize=24, framealpha=1.0)
            else:
                ax.plot(y_true, color='red', linewidth=2.5, zorder=1)
                ax.plot(y_est, color='blue', linestyle='--', linewidth=2.5, zorder=2)
            

            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.pause(1e-3)
    
    mean_errors = []
    std_errors = []
    for i, archetypes_at_level in enumerate(archetypes_perturbed):
        errors_at_level = []
        for archetype in archetypes_at_level:
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
                                  maxiter=1000
                                  ,
                                  retall=True)
            x_true = true_neuron_positions[0, :]
            error = np.linalg.norm(x_true - xf)
            errors_at_level.append(error)
            
        mean_errors.append(np.mean(errors_at_level))
        std_errors.append(np.std(errors_at_level))
    
    mean_errors = np.array(mean_errors)
    
    x = 1000*noise_levels
    y = mean_errors
    
    F = plt.figure()
    ax = F.add_subplot(1, 1, 1)
    ax.plot(x, y, linestyle='-', marker='o', color='black', linewidth=1.5, markersize=3.0)
    ax.errorbar(x, y, yerr=std_errors)
    ax.tick_params(axis='both', which='both', labelsize=24)
    
    ax.set_xlabel('Standard deviation of waveform-corrupting noise (uV)', fontsize=28)
    ax.set_ylabel('Error of neuron localization result (um)', fontsize=28)
    break