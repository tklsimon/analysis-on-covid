import scipy
import numpy as np
import matplotlib.pyplot as plt

from typing import List
from sklearn import metrics


def composite_function(f, g): 
    return lambda *x : f(g(*x))

def plot_fitted_result(predict_count_lists, true_I_counts):
    fig = plt.figure(figsize=(12, 4))
    for counts in predict_count_lists:
        plt.plot(counts)
    plt.plot(np.arange(0, len(true_I_counts)), true_I_counts, "k*:")
    plt.grid("True")
    plt.legend(["Susceptible","Infected","Removed","Original Data"])
    plt.plot()


# Compartmental Model (Base Class)
class Compartmental_model:
    def __init__(self,
                 I_counts: List[int],
                 initial_counts: List[int],
                 I_pos: int = 1):
        self.I_counts = I_counts
        self.initial_counts = initial_counts
        self.I_pos = I_pos
        self.t_length = len(I_counts)
    
    def DE(initial_counts: List[int], t, *params: List[float]):
        pass

    @staticmethod
    def cost_function(predicted: List[int], actual: List[int]):
        return sum((predicted-actual)**2)
    
    def calculate_cost(self, predicted_I: List[int]):
        return self.cost_function(predicted_I, self.I_counts)
    
    def solve(self, params: List[float], initial_counts: List[int], t_length: int, pos: int =None):
        sol = scipy.integrate.odeint(self.DE, initial_counts, range(0, t_length), args=tuple(params)).T
        if pos:
            return sol[pos]
        else:
            return sol
    
    def get_optm_params(self):
        sol = scipy.optimize.minimize(
            composite_function(self.calculate_cost, self.solve),
            self.init_params,
            (self.initial_counts, self.t_length, self.I_pos),
            method='Nelder-Mead')
        return sol.x
    
    def fit(self):
        return self.solve(self.get_optm_params(), self.initial_counts, self.t_length)

# SIR Model (Child Class)
class SIR_model(Compartmental_model):
    def __init__(self, I_counts: List[int], initial_counts: List[int], init_params=None):
        super().__init__(I_counts, initial_counts)
        if init_params:
            self.init_params = init_params
        else:
            self.init_params = (0.001, 1)
    
    @staticmethod
    def DE(initial_counts: List[int], t, *params):
        if len(initial_counts) != 3:
            raise Exception('Length of initial_counts should be 3.')
        if len(params) != 2:
            raise Exception('Number of parameters should be 2.')
        S, I, R = initial_counts[:3]
        beta, gamma = params[:2]
        
        d_S = -beta*S*I
        d_I = beta*S*I-gamma*I
        d_R = gamma*I
        
        return np.array([d_S, d_I, d_R])

# SIS Model (Child Class)
class SIS_model(Compartmental_model):
    def __init__(self, I_counts: List[int], initial_counts: List[int], init_params=None):
        super().__init__(I_counts, initial_counts)
        if init_params:
            self.init_params = init_params
        else:
            self.init_params = (0.001, 1)
    
    @staticmethod
    def DE(initial_counts, t, *params):
        if len(initial_counts) != 2:
            raise Exception('Length of initial_counts should be 2.')
        if len(params) != 2:
            raise Exception('Number of parameters should be 2.')
        S, I = initial_counts[:2]
        beta, gamma = params[:2]
        
        d_S = -beta*S*I + gamma*I
        d_I = beta*S*I - gamma*I
        
        return np.array([d_S, d_I])

# SIRS Model (Child Class)
class SIRS_model(Compartmental_model):
    def __init__(self, I_counts: List[int], initial_counts: List[int], init_params=None):
        super().__init__(I_counts, initial_counts)
        if init_params:
            self.init_params = init_params
        else:
            self.init_params = (0.001, 1)
    
    @staticmethod
    def DE(initial_counts, t, *params):
        if len(initial_counts) != 3:
            raise Exception('Length of initial_counts should be 3.')
        if len(params) != 3:
            raise Exception('Number of parameters should be 3.')
        S, I, R = initial_counts[:3]
        beta, gamma, sigma = params[:3]
        
        d_S = -beta*S*I + sigma*R
        d_I = beta*S*I - gamma*I
        d_R = gamma*I - sigma*R
        
        return np.array([d_S, d_I, d_R])

# SEIR Model (Child Class)
class SEIR_model(Compartmental_model):
    def __init__(self, I_counts: List[int], initial_counts: List[int], init_params=None):
        super().__init__(I_counts, initial_counts, 2)
        if init_params:
            self.init_params = init_params
        else:
            self.init_params = (0.001, 1)
    
    @staticmethod
    def DE(initial_counts, t, *params):
        if len(initial_counts) != 4:
            raise Exception('Length of initial_counts should be 4.')
        if len(params) != 3:
            raise Exception('Number of parameters should be 3.')
        S, E, I, R = initial_counts[:4]
        beta, aleph, gamma = params[:3]
        
        d_S = -beta*S*I
        d_E = beta*S*I - aleph*E
        d_I = aleph*E - gamma*I
        d_R = gamma*I
        
        return np.array([d_S, d_E, d_I, d_R])