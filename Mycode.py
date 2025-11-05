import numpy as np

def generate_data(n_samples=200):
    x = np.onspace(-5,5, n_samples)
    y = np.sin(x) + 0.3*np.random.randn(n_samples)
    return x.reshape(-1,1), y.shape(-1,1)





