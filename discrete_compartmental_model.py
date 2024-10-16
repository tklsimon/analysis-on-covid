import scipy
import numpy as np
import matplotlib.pyplot as plt


def composite_function(f, g): 
    return lambda *x : f(g(*x))

def plot_compartmental_model_result(index_x_dt,
                                    fitted_model,
                                    true_plus_list):
    line_color_dict = {'Susceptible': 'b',
                       'Exposed': 'm',
                       'Infectious': 'r',
                       'Recovered': 'g'}
    fig = plt.figure(figsize=(12, 4))
    predicted_plus_list, compartment_counts_list = fitted_model.output
    for counts, class_name in zip(compartment_counts_list, fitted_model.classes):
        plt.plot(index_x_dt, counts, line_color_dict[class_name])
    plt.plot(index_x_dt, predicted_plus_list[fitted_model.classes.index('Infectious')-1], 'r--')
    plt.plot(index_x_dt, true_plus_list[0], 'k*:')
    plt.title('Compartmental Model Result')
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.grid('True')
    plt.legend(fitted_model.classes + ['Predicted new case counts'] + ['True new case counts'])
    plt.plot()

# Discrete Compartmental Model (Base Class)
class Discrete_Compartmental_model:
    def __init__(self,
                 init_params):
        self.init_params = init_params
    
    @staticmethod
    def cost_function(predicted, true):
        cost = (predicted-true)**2
        for _ in range(len(cost.shape)):
            cost = sum(cost)
        return cost
    
    def set_true_plus_list(self,
                           true_plus_list):
        self.true_plus_list = true_plus_list
    
    def calculate_cost(self, predicted_DI_plus_list):
        return self.cost_function(predicted_DI_plus_list, self.true_plus_list)
    
    def solve(params,
              initial_counts,
              t_length,
              output=None):
        pass
    
    def get_optm_params(self,
                        true_plus_list,
                        initial_counts):
        if true_plus_list.shape[0] == 1:
            model_output = 'DI_plus_list'
        elif true_plus_list.shape[0] > 1:
            model_output = 'plus_list'
            
        self.set_true_plus_list(true_plus_list)
        
        sol = scipy.optimize.minimize(
            composite_function(self.calculate_cost, self.solve),
            self.init_params,
            (initial_counts, true_plus_list.shape[1], model_output),
            method='Nelder-Mead')
        
        return sol.x
    
    def fit(self,
            true_plus_list,
            initial_counts):
        print('Fitting compartmental model ...')
        optm_params = self.get_optm_params(true_plus_list, initial_counts)
        print('The optimal parameters found are:', optm_params)
        print('Generating the counts as model output ...')
        self.output = self.solve(optm_params, initial_counts, true_plus_list.shape[1])
        print('Completed.')

# Discrete SIR Model (Child Class)
class Discrete_SIR_model(Discrete_Compartmental_model):
    def __init__(self,
                 init_params=(8e-1, 1e-1)):
        super().__init__(init_params)
        self.classes = ['Susceptible', 'Infectious', 'Recovered']

    @staticmethod
    def solve(params,
              initial_counts,
              t_length,
              output=None):
        S_0, I_0, R_0 = initial_counts
        N = sum(initial_counts)
        beta, gamma = params
    
        beta = min(max(beta, 1e-5), 10)
        gamma = min(max(gamma, 1e-5), 1)
    
        S_list, I_list, R_list = [S_0], [I_0], [R_0]
        DI_plus_list, DR_plus_list = [I_0], [R_0]
        for _ in range(t_length-1):
            DI_plus = int(beta*S_list[-1]*I_list[-1]/N)
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
            return DI_plus_list
        elif output == 'plus_list':
            return plus_list
        elif output == 'count_list':
            return count_list
        else:
            return [plus_list, count_list]

# Discrete SIS Model (Child Class)
class Discrete_SIS_model(Discrete_Compartmental_model):
    def __init__(self,
                 init_params=(8e-1, 5e-2)):
        super().__init__(init_params)
        self.classes = ['Susceptible', 'Infectious']

    @staticmethod
    def solve(params,
              initial_counts,
              t_length,
              output=None):
        S_0, I_0 = initial_counts
        N = sum(initial_counts)
        beta, gamma = params
    
        beta = min(max(beta, 1e-5), 10)
        gamma = min(max(gamma, 1e-5), 1)
    
        S_list, I_list = [S_0], [I_0]
        DI_plus_list, DS_plus_list = [I_0], [0]
        for _ in range(t_length-1):
            DI_plus = int(beta*S_list[-1]*I_list[-1]/N)
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
            return DI_plus_list
        elif output == 'plus_list':
            return plus_list
        elif output == 'count_list':
            return count_list
        else:
            return [plus_list, count_list]


# Discrete SIRS Model (Child Class)
class Discrete_SIRS_model(Discrete_Compartmental_model):
    def __init__(self,
                 init_params=(8e-1, 1e-1, 1e-2)):
        super().__init__(init_params)
        self.classes = ['Susceptible', 'Infectious', 'Recovered']

    @staticmethod
    def solve(params,
              initial_counts,
              t_length,
              output=None):
        S_0, I_0, R_0 = initial_counts
        N = sum(initial_counts)
        beta, gamma, sigma = params
    
        beta = min(max(beta, 1e-5), 10)
        gamma = min(max(gamma, 1e-5), 1)
        sigma = min(max(sigma, 1e-5), 1)
    
        S_list, I_list, R_list = [S_0], [I_0], [R_0]
        DI_plus_list, DR_plus_list, DS_plus_list = [I_0], [R_0], [0]
        for _ in range(t_length-1):
            DI_plus = int(beta*S_list[-1]*I_list[-1]/N)
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
            return DI_plus_list
        elif output == 'plus_list':
            return plus_list
        elif output == 'count_list':
            return count_list
        else:
            return [plus_list, count_list]


# Discrete SEIR Model (Child Class)
class Discrete_SEIR_model(Discrete_Compartmental_model):
    def __init__(self,
                 init_params=(8e-1, 5e-1, 1e-1)):
        super().__init__(init_params)
        self.classes = ['Susceptible', 'Exposed', 'Infectious', 'Recovered']

    @staticmethod
    def solve(params,
              initial_counts,
              t_length,
              output=None):
        S_0, E_0, I_0, R_0 = initial_counts
        N = sum(initial_counts)
        beta, alpha, gamma = params
    
        beta = min(max(beta, 1e-5), 10)
        alpha = min(max(alpha, 1e-5), 1)
        gamma = min(max(gamma, 1e-5), 1)
    
        S_list, E_list, I_list, R_list = [S_0], [E_0], [I_0], [R_0]
        DE_plus_list, DI_plus_list, DR_plus_list = [E_0], [I_0], [R_0]
        for _ in range(t_length-1):
            DE_plus = int(beta*S_list[-1]*I_list[-1]/N)
            DI_plus = int(alpha*E_list[-1])
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
            return DI_plus_list
        elif output == 'plus_list':
            return plus_list
        elif output == 'count_list':
            return count_list
        else:
            return [plus_list, count_list]
