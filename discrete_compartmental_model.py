import scipy
import numpy as np
import matplotlib.pyplot as plt


def composite_function(f, g): 
    return lambda *x : f(g(*x))

def plot_compartmental_model_result(model_output,
                                    true_plus_list,
                                    classes):
    line_color_dict = {'Susceptible': 'b',
                       'Exposed': 'm',
                       'Infectious': 'r',
                       'Recovered': 'g'}
    fig = plt.figure(figsize=(12, 4))
    predicted_plus_list, compartment_counts_list = model_output
    for counts, class_name in zip(compartment_counts_list, classes):
        plt.plot(counts, line_color_dict[class_name])
    plt.plot(predicted_plus_list[classes.index('Infectious')-1], 'r--')
    plt.plot(true_plus_list[0], 'k*:')
    plt.title('Compartmental Model Result')
    plt.xlabel('Days')
    plt.ylabel('Count')
    plt.grid('True')
    plt.legend(classes + ['Predicted case counts'] + ['True case counts'])
    plt.plot()

# Discrete Compartmental Model (Base Class)
class Discrete_Compartmental_model:
    def __init__(self,
                 true_plus_list,
                 initial_counts,
                 init_params):
        self.true_plus_list = true_plus_list
        self.initial_counts = initial_counts
        self.init_params = init_params
        self.true_plus_list_nbr, self.t_length = self.true_plus_list.shape
    
    @staticmethod
    def cost_function(predicted, true):
        cost = (predicted-true)**2
        for _ in range(len(cost.shape)):
            cost = sum(cost)
        return cost
    
    def calculate_cost(self, predicted_DI_plus_list):
        return self.cost_function(predicted_DI_plus_list, self.true_plus_list)
    
    def solve(params,
              initial_counts,
              t_length,
              output=None):
        pass
    
    def get_optm_params(self):
        if self.true_plus_list_nbr == 1:
            model_output = 'DI_plus_list'
        elif self.true_plus_list_nbr > 1:
            model_output = 'plus_list'
        sol = scipy.optimize.minimize(
            composite_function(self.calculate_cost, self.solve),
            self.init_params,
            (self.initial_counts, self.t_length, model_output),
            method='Nelder-Mead')
        return sol.x
    
    def fit(self):
        optm_params = self.get_optm_params()
        print('The optimal parameters are:', optm_params)
        return self.solve(optm_params, self.initial_counts, self.t_length)

# Discrete SIR Model (Child Class)
class Discrete_SIR_model(Discrete_Compartmental_model):
    def __init__(self,
                 true_plus_list,
                 initial_counts,
                 init_params=(0.001, 0.1)):
        super().__init__(true_plus_list, initial_counts, init_params)
        self.classes = ['Susceptible', 'Infectious', 'Recovered']

    @staticmethod
    def solve(params,
              initial_counts,
              t_length,
              output=None):
        S_0, I_0, R_0 = initial_counts
        beta, gamma = params
    
        beta = min(max(beta, 0), 1)
        gamma = min(max(gamma, 0), 1)
    
        S_list, I_list, R_list = [S_0], [I_0], [R_0]
        DI_plus_list, DR_plus_list = [I_0], [R_0]
        for _ in range(t_length-1):
            DI_plus = int(beta*S_list[-1]*I_list[-1])
            DR_plus = int(gamma*I_list[-1])
    
            if DI_plus > S_list[-1]:
                DI_plus = S_list[-1]
            if DR_plus > I_list[-1] + DI_plus:
                DR_plus = I_list[-1] + DI_plus
            
            DI_plus_list.append(DI_plus)
            DR_plus_list.append(DR_plus)
            S_list.append(S_list[-1] - DI_plus)
            I_list.append(I_list[-1] + DI_plus - DR_plus)
            R_list.append(R_list[-1] + DR_plus)

        plus_list = [DI_plus_list, DR_plus_list]
        count_list = [S_list, I_list, R_list]
        if output == 'DI_plus_list':
            return plus_list[0]
        elif output == 'plus_list':
            return plus_list
        elif output == 'count_list':
            return count_list
        else:
            return [plus_list, count_list]

# Discrete SIS Model (Child Class)
class Discrete_SIS_model(Discrete_Compartmental_model):
    def __init__(self,
                 true_plus_list,
                 initial_counts,
                 init_params=(0.001, 0.00001)):
        super().__init__(true_plus_list, initial_counts, init_params)
        self.classes = ['Susceptible', 'Infectious']

    @staticmethod
    def solve(params,
              initial_counts,
              t_length,
              output=None):
        S_0, I_0 = initial_counts
        beta, gamma = params
    
        beta = min(max(beta, 0), 1)
        gamma = min(max(gamma, 0), 1)
    
        S_list, I_list = [S_0], [I_0]
        DI_plus_list, DS_plus_list = [I_0], [0]
        for _ in range(t_length-1):
            DI_plus = int(beta*S_list[-1]*I_list[-1])
            DS_plus = int(gamma*I_list[-1])
    
            if DI_plus > S_list[-1]: # Not S_list[-1] + DS_plus, to avoid circular update issue
                DI_plus = S_list[-1]
            if DS_plus > I_list[-1] + DI_plus:
                DS_plus = I_list[-1] + DI_plus
            
            DI_plus_list.append(DI_plus)
            DS_plus_list.append(DS_plus)
            S_list.append(S_list[-1] - DI_plus + DS_plus)
            I_list.append(I_list[-1] + DI_plus - DS_plus)

        plus_list = [DI_plus_list, DS_plus_list]
        count_list = [S_list, I_list]
        if output == 'DI_plus_list':
            return plus_list[0]
        elif output == 'plus_list':
            return plus_list
        elif output == 'count_list':
            return count_list
        else:
            return [plus_list, count_list]


# Discrete SIRS Model (Child Class)
class Discrete_SIRS_model(Discrete_Compartmental_model):
    def __init__(self,
                 true_plus_list,
                 initial_counts,
                 init_params=(0.001, 0.2, 0.1)):
        super().__init__(true_plus_list, initial_counts, init_params)
        self.classes = ['Susceptible', 'Infectious', 'Recovered']

    @staticmethod
    def solve(params,
              initial_counts,
              t_length,
              output=None):
        S_0, I_0, R_0 = initial_counts
        beta, gamma, sigma = params
    
        beta = min(max(beta, 0), 1)
        gamma = min(max(gamma, 0), 1)
        sigma = min(max(sigma, 0), 1)
    
        S_list, I_list, R_list = [S_0], [I_0], [R_0]
        DI_plus_list, DR_plus_list, DS_plus_list = [I_0], [R_0], [0]
        for _ in range(t_length-1):
            DI_plus = int(beta*S_list[-1]*I_list[-1])
            DR_plus = int(gamma*I_list[-1])
            DS_plus = int(sigma*R_list[-1])
    
            if DI_plus > S_list[-1]:
                DI_plus = S_list[-1]
            if DR_plus > I_list[-1] + DI_plus:
                DR_plus = I_list[-1] + DI_plus
            if DS_plus > R_list[-1] + DR_plus:
                DS_plus = R_list[-1] + DR_plus
            
            DI_plus_list.append(DI_plus)
            DR_plus_list.append(DR_plus)
            DS_plus_list.append(DS_plus)
            S_list.append(S_list[-1] - DI_plus + DS_plus)
            I_list.append(I_list[-1] + DI_plus - DR_plus)
            R_list.append(R_list[-1] + DR_plus - DS_plus)

        plus_list = [DI_plus_list, DR_plus_list, DS_plus_list]
        count_list = [S_list, I_list, R_list]
        if output == 'DI_plus_list':
            return plus_list[0]
        elif output == 'plus_list':
            return plus_list
        elif output == 'count_list':
            return count_list
        else:
            return [plus_list, count_list]


# Discrete SEIR Model (Child Class)
class Discrete_SEIR_model(Discrete_Compartmental_model):
    def __init__(self,
                 true_plus_list,
                 initial_counts,
                 init_params=(0.0005, 0.8, 0.1)):
        super().__init__(true_plus_list, initial_counts, init_params)
        self.classes = ['Susceptible', 'Exposed', 'Infectious', 'Recovered']

    @staticmethod
    def solve(params,
              initial_counts,
              t_length,
              output=None):
        S_0, E_0, I_0, R_0 = initial_counts
        beta, aleph, gamma = params
    
        beta = min(max(beta, 0), 1)
        aleph = min(max(aleph, 0), 1)
        gamma = min(max(gamma, 0), 1)
    
        S_list, E_list, I_list, R_list = [S_0], [E_0], [I_0], [R_0]
        DE_plus_list, DI_plus_list, DR_plus_list = [E_0], [I_0], [R_0]
        for _ in range(t_length-1):
            DE_plus = int(beta*S_list[-1]*I_list[-1])
            DI_plus = int(aleph*E_list[-1])
            DR_plus = int(gamma*I_list[-1])
    
            if DE_plus > S_list[-1]:
                DE_plus = S_list[-1]
            if DI_plus > E_list[-1] + DE_plus:
                DI_plus = E_list[-1] + DE_plus
            if DR_plus > I_list[-1] + DI_plus:
                DR_plus = I_list[-1] + DI_plus
            
            DE_plus_list.append(DE_plus)
            DI_plus_list.append(DI_plus)
            DR_plus_list.append(DR_plus)
            S_list.append(S_list[-1] - DE_plus)
            E_list.append(E_list[-1] + DE_plus - DI_plus)
            I_list.append(I_list[-1] + DI_plus - DR_plus)
            R_list.append(R_list[-1] + DR_plus)
        
        plus_list = [DE_plus_list, DI_plus_list, DR_plus_list]
        count_list = [S_list, E_list, I_list, R_list]
        if output == 'DI_plus_list':
            return plus_list[1]
        elif output == 'plus_list':
            return plus_list
        elif output == 'count_list':
            return count_list
        else:
            return [plus_list, count_list]