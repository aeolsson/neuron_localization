import numpy as np
from sklearn.decomposition import FastICA

import params

class Spike_detector:
    def __init__(self,
                 threshold,
                 polarity,
                 refractory_period,
                 extraction_window,
                 alignment_window,
                 preprocessing=None):
        
        self.polarity = polarity
        self.threshold = threshold
        self.alignment_window = alignment_window
        self.refractory_period = refractory_period
        self.extraction_window = extraction_window
        
        self.preprocessing = preprocessing
    
    def get_waveforms(self, signal, spike_times):
        waveforms = []
        
        for i, spike_time in enumerate(list(spike_times)):
            waveform = signal[:, spike_time-self.extraction_window[0]:spike_time+self.extraction_window[1]+1]
            waveforms.append(waveform)
        
        return waveforms
    
    def detect_spike_train(self, signal):
        spike_times, threshold = self.detect_spike_times(signal)
        spike_times = np.array(spike_times)
        
        spike_train = np.zeros(np.max(signal.shape))
        spike_train[spike_times] = 1.0
        
        return spike_train, spike_times, threshold
    
    def detect_spike_times(self, signal):
        if self.preprocessing:
            signal = self.preprocessing(signal)
        
        MAD = np.median(np.abs(signal - np.median(signal)))
        threshold = self.threshold * MAD
        signal_thres = signal - threshold
        
        detector_signal = np.zeros_like(signal)
        detector_signal[self.polarity * signal_thres > 0] = 1.0
        
        detector_signal = np.diff(detector_signal)
        detector_signal = np.insert(detector_signal, obj=0, values=0.0)
        detector_signal[detector_signal < 0] = 0
        
        spike_times = list(np.where(detector_signal)[0])
        
        if len(spike_times)==0:
            return [], threshold
        
        spike_times = self.enforce_refractory_period(spike_times)
        
        spike_times = self.align(spike_times=spike_times, signal=signal)
        return spike_times, threshold
    
    def align(self, spike_times, signal):
        spike_times = np.array(spike_times)
        
        signal_duration = np.size(signal)
        
        spike_times = spike_times[spike_times + self.extraction_window[1] < signal_duration]
        
        spike_times_aligned = np.zeros_like(spike_times)
        
        for i in range(spike_times.size):
            if spike_times[i] + self.alignment_window <= signal_duration:
                iwin = np.arange(spike_times[i], spike_times[i] + self.alignment_window)
            else:
                iwin = np.arange(spike_times[i], signal_duration)
            
            if self.polarity == 1:
                imax = np.argmax(signal[iwin])
                
            elif self.polarity == -1:
                imax = np.argmin(signal[iwin])
            
            spike_times_aligned[i] = spike_times[i] + imax
        
        spike_times_aligned = list(spike_times_aligned)
        return spike_times_aligned
    
    def enforce_refractory_period(self, spike_times):
        spike_times_ok = [spike_times[0]]
        
        for i in range(1, len(spike_times)):
            if spike_times[i] - spike_times_ok[-1] > self.refractory_period:
                spike_times_ok.append(spike_times[i])
        
        return spike_times_ok
        

class Single_channel_spike_detector(Spike_detector):
    
    def __init__(self,
                 polarity,
                 threshold,
                 alignment_window=10,
                 refractory_period=24,
                 extraction_window=[23, 76],
                 preprocessing=None):
        self.polarity = polarity
        
        super().__init__(threshold=threshold,
                         alignment_window=alignment_window,
                         refractory_period=refractory_period,
                         extraction_window=extraction_window,
                         preprocessing=preprocessing)


class Multichannel_spike_detector(Spike_detector):
    
    def __init__(self,
                 polarity=-1,
                 threshold=0,
                 extraction_window=[24, 75],
                 alignment_window=10,
                 refractory_period=24,
                 preprocessing=None):
        
        super().__init__(polarity=polarity,
                         threshold=threshold,
                         alignment_window=alignment_window,
                         refractory_period=refractory_period,
                         extraction_window=extraction_window,
                         preprocessing=preprocessing)
    
    def detect_spike_times(self, signals):
        num_channels = signals.shape[0]
        
        spike_times = []
        thresholds = []
        for channel in range(num_channels):
            signal = signals[channel, :]
            
            channel_spike_times, channel_threshold = super().detect_spike_times(signal)
            
            spike_times += channel_spike_times
            thresholds.append(channel_threshold)
        
        spike_times.sort()
        spike_times = self.enforce_refractory_period(spike_times)
        return spike_times, thresholds
        
    
#    def demix(self, signals):
##        num_channels = signals.shape[0]
##        num_samples = signals.shape[1]
##        waveform_length = sum(self.extraction_window)+1
##        sources, peeled_sources, spike_trains, indices = emgDecomposition.peel_off_ica(signals.T.copy(),
##                                                                                       params.fs,
##                                                                                       iterations=self.num_sources,
##                                                                                       extension=waveform_length)
##        sources = sources.T
##        sources = (sources.T / np.max(np.abs(sources), axis=1)).T
##        return sources, spike_trains
#        
#        num_channels = signals.shape[0]
#        num_samples = signals.shape[1]
#        waveform_length = sum(self.extraction_window)+1
#        
#        signals = np.copy(signals)
#        sources = []
#        spike_trains = []
#        for s in range(self.num_sources):
#            signals_rep = self.extend(signals, factor=waveform_length)
#            
#            demixer = FastICA(n_components=waveform_length,
#                              algorithm='parallel',
#                              whiten=True,
#                              fun='logcosh',
#                              max_iter=500)
#            source = demixer.fit_transform(signals_rep.T).T
#            source = source[0, :]
#            source = source / np.max(np.abs(source))
#            
#            spike_train = (np.abs(source) > self.threshold).astype(np.float64)
#            
#            signals = self.remove_spikes(signals, spike_train)
#            
#            sources.append(source)
#            spike_trains.append(spike_train)
#        
#        sources = np.array(sources)
#        spike_trains = np.array(spike_trains)
#        return sources, spike_trains
#    
#    def extend(self, signals, factor):
#        num_channels = signals.shape[0]
#        num_samples = signals.shape[1]
#        
#        signals_rep = np.zeros([num_channels*factor, num_samples])
#        for channel in range(num_channels):
#                for offset in range(factor):
#                    offset_signal = np.roll(signals[channel, :], offset)
#                    offset_signal[:offset:] = 0.0
#                    signals_rep[channel+offset, :] = offset_signal
#        
#        return signals_rep
#    
#    def remove_spikes(self, signals, spike_train):
#        spike_times = np.where(spike_train)[0]
#        signals = signals.copy()
#        
#        waveform_length = sum(self.extraction_window)+1
#        
#        for spike_time in spike_times:
#            dead_zone = slice(spike_time - waveform_length, spike_time + waveform_length + 1, 1)
#            signals[:, dead_zone] /= 100.0
#            
#        return signals
#        