import numpy as np

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class Spike_sorter:
    
    def __init__(self):
        pass
    
class PCA_spike_sorter(Spike_sorter):
    
    def __init__(self, num_sources, num_components=10):
        self.pca = PCA(n_components=num_components)
        self.kmeans = KMeans(n_clusters=num_sources)
    
    def sort(self, waveforms):
        X = []
        for waveform in waveforms:
            X.append(waveform.flatten())
        X = np.array(X)
        Y = self.pca.fit_transform(X)
        self.kmeans.fit(X)
        C = self.kmeans.predict(X)
        return C, Y