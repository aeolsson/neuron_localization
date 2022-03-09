import numpy as np
from scipy.signal import resample

import params
import utils

class Waveform_model:
    
    def __init__(self,
                 PC,
                 DB,
                 k_mat,
                 model_volume,
                 far_field_amp,
                 fs,
                 x=0,
                 y=0,
                 z=0):
        
        self.PC = PC
        self.DB = DB
        self.k_mat = k_mat
        self.model_volume = model_volume
        self.far_field_amp = far_field_amp
        self.fs = fs
        self.x = x
        self.y = y
        self.z = z
    
    @classmethod
    def from_mat(cls, path, x=0, y=0, z=0):
        mat = utils.loadmat(path)['neuronModel']
        PC = mat['PC']
        DB = mat['DB']
        k_mat = mat['kMat']
        model_volume = mat['modelVolume']
        far_field_amp = mat['farFieldAmp']
        fs = mat['fs']
        
        return cls(PC=PC,
                   DB=DB,
                   k_mat=k_mat,
                   model_volume=model_volume,
                   far_field_amp=far_field_amp,
                   fs=fs,
                   x=x,
                   y=y,
                   z=z)
    
    def get_waveform(self,
                 x=0,
                 y=0,
                 z=0):
        
        # Compute position of electrode in coordinate system of neuron
        rel_pos = np.array([x - self.x,
                            y - self.y,
                            z - self.z])
        
        # Check if electrode is inside/outside model volume (ellipsoid)
        rtest = np.sqrt(rel_pos[0]**2 / self.model_volume[0]**2 + \
                        rel_pos[1]**2 / self.model_volume[1]**2 + \
                        rel_pos[2]**2 / self.model_volume[2]**2)
        
        # Scaling of coordinates
        k = 1 if rtest <= 1 else 1 / rtest
        k = np.array([k, k, k])
        xyz = k * rel_pos
        
        # Distance to ellipsoid
        d = 0 if rtest <= 1 else np.sqrt(np.sum((xyz - rel_pos) ** 2))
        
        # Amplitude attenuation
        damp = 1 / (1 + self.far_field_amp[0]*d) ** self.far_field_amp[1]
        
        # Construct Vandermonde matrix
        A = np.ones([1, self.k_mat.shape[0]])
        for i in range(3):
            update = [xyz[i] ** self.k_mat[j, i] for j in range(self.k_mat.shape[0])]
            update = np.array(update)
            A = A * update
        
        waveform = damp * self.PC @ (A @ self.DB).T
        waveform = waveform[:, 0]
        
        waveform = resample(x=waveform, num=params.waveform_duration)
        
        return waveform
           