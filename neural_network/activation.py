import numpy as np

def activation_sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def activation_tanh(x):
    return np.tanh(x)

def activation_atan(x):
    return np.atan(x)

def activation_identity(x):
    return x

activation_functions = {'sigmoid': activation_sigmoid, 'tanh': activation_tanh, 'atan': activation_atan, 'identity': activation_identity}

