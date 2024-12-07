import numpy as np
from scipy.integrate import odeint

def ode_model(x, t):
    return [x[1], -x[0] + (1 - x[0]**2)*x[1]]

def Phi(x):
    t = np.linspace(0, 0.05, 101)
    sol = odeint(ode_model, x, t)
    return sol[-1]

def generate_data(n : int, start : float, end : float):
    sample_space = np.linspace(start, end, n)
    X = np.array(np.meshgrid(sample_space, sample_space)).T.reshape(-1, 2)
    Y = np.apply_along_axis(Phi, 1, X)

    return X, Y