import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR, VARResults
from typing import List, Union

from covid_module import print_or_not


def stationary_and_difference(df: pd.DataFrame,
                              col_list: List[str]=None,
                              print_ind: bool=False) -> None:
    if col_list is None:
        col_list = df.columns
    
    new_col_list = []
    for col in col_list:
        p_value = adfuller(df[col])[1]
        if p_value > 0.05:
            print_or_not(f'The column {col} has ADF p-value {p_value:.5f} which is non-stationary.', print_ind)
            
            col_diff = col + '_diff_1'
            if '_diff_' in col:
                col_list = col.split('_diff_')
                try:
                    col_list[-1] = str(int(col_list[-1]) + 1)
                    col_diff = '_diff_'.join(col_list)
                except:
                    pass
            
            print_or_not(f'--> Replacing the column {col} with its difference column {col_diff} ...', print_ind)
            df[col_diff] = df[col].diff().fillna(0)
            df.drop(col, axis=1, inplace=True)
            new_col_list.append(col_diff)
        else:
            print_or_not(f'The column {col} has ADF p-value {p_value:.5f} which is stationary.', print_ind)
    
    print_or_not('')
    return new_col_list


def stationary_and_difference_loop(df: pd.DataFrame,
                                   col_list: List[str]=None,
                                   max_iter: int=np.inf,
                                   print_ind: bool=False) -> None:
    if col_list is None:
        col_list = df.columns
    
    nbr = 1
    print('Running the function stationary_and_difference using for loop ...\n')
    while True:
        if len(col_list) == 0:
            print('All target columns are now stationary.')
            break
        if nbr > max_iter:
            print(f'Maximum iteration {max_iter} reached. Stopped.')
            break
        print(f'Running Loop #{nbr} ...')
        col_list = stationary_and_difference(df, col_list, print_ind)
        nbr += 1
    print('Loop Ended.')


def print_model_result(summary_str: str,
                       y_col_list: List[str]) -> None:
    summary_split_list = summary_str.split('\n')
    summary_result_index = [index for index, str in enumerate(summary_split_list) if \
                                str.startswith('Results for equation') or str.startswith('Correlation matrix of residuals')]
    
    for y_col in y_col_list:
        for i in range(len(summary_result_index)):
            if y_col in summary_split_list[summary_result_index[i]]:
                print('\n'.join(summary_split_list[summary_result_index[i]:summary_result_index[i+1]]))


def fit_var_model(var_data: pd.DataFrame,
                  X_col_list: Union[List[str], str],
                  y_col_list: Union[List[str], str],
                  print_result_ind: bool=False) -> VARResults:
    if isinstance(X_col_list, str):
        X_col_list = [X_col_list]
    if isinstance(y_col_list, str):
        y_col_list = [y_col_list]
        
    X_col_list = [col for col in var_data.columns if col.split('_diff_')[0] in X_col_list]
    y_col_list = [col for col in var_data.columns if col.split('_diff_')[0] in y_col_list]
    
    # Select required columns
    var_data = var_data[X_col_list + y_col_list]
    
    # VAR model
    var_model = VAR(var_data)
    
    # Select the maximum lag order based on AIC
    max_lag_order = var_model.select_order().aic
    
    # Fit the VAR model with selected maximum lag
    print(f'Fitting the VAR model with maximum lag {max_lag_order} ...')
    var_model = var_model.fit(maxlags=max_lag_order)
    # Save the maximum lag order to the model output
    var_model.max_lag_order = max_lag_order
    print(f'Done.\n')

    # Print model summary
    summary_str = str(var_model.summary())
    if print_result_ind:
        print_model_result(summary_str, y_col_list)
    
    return var_model

def get_significant_variable(var_model: VARResults,
                             y_col_list: Union[List[str], str],
                             p_val_thrhld: float) -> List[str]:
    if isinstance(y_col_list, str):
        y_col_list = [y_col_list]
    
    var_model_pvalues_df = var_model.pvalues
    
    y_col_list = [col for col in var_model_pvalues_df.columns \
                      if col.split('_diff_')[0] in y_col_list]

    signf_cols_list = []
    for y_col in y_col_list: 
        print(f'For {y_col},')
        pvalues_df_temp = var_model_pvalues_df \
            [(var_model_pvalues_df[y_col] < p_val_thrhld) & (var_model_pvalues_df.index != 'const')] \
            [y_col]
        
        signf_cols_list_temp = pvalues_df_temp.index.to_list()
        signf_pvalues_list_temp = pvalues_df_temp.values.tolist()

        if len(signf_cols_list_temp) > 0:
            print(f'The following variables are significant (p-value < {p_val_thrhld}):')
            print('(variable name: p-value)')
            for col, p_value in zip(signf_cols_list_temp, signf_pvalues_list_temp):
                print(f'{col}: {p_value:.3}')
        else:
            print('There are no significant variables.')
        print('')
        
        signf_cols_list.extend(signf_cols_list_temp)
    return signf_cols_list


def add_lagged_column(df: pd.DataFrame,
                      lag_col_list: List[str]) -> pd.DataFrame:
    df_lag = df.copy()
    
    for col in lag_col_list:
        col_part_list = col.split('.', 1)
        
        if col_part_list[0][0] == 'L':
            try:
                lag_nbr = int(col_part_list[0][1:])
                col = col_part_list[1]
                df_lag[f'L{lag_nbr}.{col}'] = df_lag[col].shift(lag_nbr).fillna(0)
            except:
                pass
    
    return df_lag
