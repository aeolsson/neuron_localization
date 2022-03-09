import numpy as np
import scipy.io as sio
import scipy.signal as ssignal

def _check_keys( dict):
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def loadmat(filename):
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def bandpass(signal, fbp, fs):
    Blp,Alp = ssignal.bessel(2,1, analog=True)
    Blp,Alp = ssignal.lp2lp(Blp,Alp,fbp[1]*2*np.pi)
    blp,alp = ssignal.bilinear(Blp,Alp,fs)
    
    Bhp,Ahp = ssignal.bessel(4,1, analog=True)
    Bhp,Ahp = ssignal.lp2hp(Bhp,Ahp,fbp[0]*2*np.pi)
    bhp,ahp = ssignal.bilinear(Bhp,Ahp,fs)
    
    vlp = ssignal.lfilter(blp,alp,signal)
    filteredSignal = ssignal.filtfilt(bhp,ahp,vlp)
    
    import matplotlib.pyplot as plt
    [flp,Hlp] = ssignal.freqz(b=blp,a=alp,worN=256,fs=fs)
    [fhp,Hhp] = ssignal.freqz(b=bhp,a=ahp,worN=256,fs=fs);
    f = plt.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.plot(flp, np.abs(Hlp))
    ax.plot(fhp, np.abs(Hhp))
    plt.draw()
    plt.pause(1e-3)
    plt.show()
    
    return filteredSignal

def times2spikes(times, length):
    times = np.array(times)
        
    spike_train = np.zeros(length)
    spike_train[times] = 1.0
    
    return spike_train