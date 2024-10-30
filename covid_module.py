import datetime
import numpy as np
import pandas as pd
import random

from typing import Union, List, Set


def print_or_not(string: str,
                 print_ind: bool=True) -> None:
    if print_ind:
        print(string)


def print_cumulative_columns(df: pd.DataFrame) -> None:
    col_list = df.columns
    nbr_col_list = [col for col in col_list if \
                        'NUMBER' in col.upper() or \
                        'CNT' in col.upper()]
    
    cuml_col_list = []
    non_cuml_col_list = []
    for col in nbr_col_list:
        if df[~df[col].isna()][col].is_monotonic:
            cuml_col_list.append(col)
        else:
            non_cuml_col_list.append(col)
    
    print('The cumulative columns in dataset are:')
    print(*cuml_col_list, sep = "\n")
    print('')
    print('The non-cumulative columns in dataset are:')
    print(*non_cuml_col_list, sep = "\n")


def print_range_of_data(df: pd.DataFrame,
                        date_col: str,
                        target_col: str) -> None:
    missing_ind = df[target_col].isna()
    df_agg_result = df[~missing_ind].groupby((missing_ind).cumsum())
    if len(df_agg_result) > 0:
        print(f'The column "{target_col}" has data between:')
        for _, df_temp in df_agg_result:
            date_list = [*df_temp[date_col].astype('str')]
            if len(date_list) > 1:
                print(f'From {date_list[0]} to {date_list[-1]}')
            else:
                print(f'{date_list[0]}')
    else:
        print(f'The column "{target_col}" does not have any data.')
    print('')


def print_missing_val_count(df: pd.DataFrame) -> None:
    # Missing value counts
    df_na_count = df.isnull().sum()
    df_record_count = df.shape[0]
    
    # Print the count of missing value for each feature
    if df_na_count.sum() > 0:
        print('The following columns have missing values:')
        for col, na_count in zip(df_na_count.index, df_na_count.values):
            if na_count > 0:
                print(f'{col}: {na_count} ({100*na_count/df_record_count:0.1f}%)')
    else:
        print('This dataframe does not have missing values.')


def change_date_format(df: pd.DataFrame,
                       old_col_name: str,
                       old_date_format: str,
                       new_col_name: str,
                       new_date_format: str) -> None:
    df[new_col_name] = \
        pd.to_datetime(df[old_col_name], format=old_date_format, errors='coerce') \
        .dt.strftime(new_date_format) \
        .fillna('N/A')


def get_year_month_part(df: pd.DataFrame,
                        col_name: str,
                        date_format: str) -> None:
    for (date_part, date_part_format) in [('year','%Y'), ('year_month','%Y%m')]:
        new_col_name = col_name.removesuffix('_date') + '_' + date_part
        change_date_format(df,
                           col_name,
                           date_format,
                           new_col_name,
                           date_part_format)


# def one_hot_encoding(df: pd.DataFrame,
#                      col_name: str) -> None:
#     distinct_values = set(df[col_name])
    
#     for value in distinct_values:
#         new_col_name = col_name + '_' + value
#         df[new_col_name] = df[col_name].apply(lambda col: 1 if col == value else 0)


def get_distance_betwn_lists(list_1: Union[List[str], Set[str]], 
                             list_2: Union[List[str], Set[str]]) -> float:
    if (len(list_1) > 0) or (len(list_2) > 0):
        similarity = len(set(list_1) & set(list_2)) / len(set(list_1) | set(list_2))
        distance = 1 - similarity
    else:
        distance = 0
    return distance

def get_distance_betwn_columns(df: pd.DataFrame,
                               col_A: str,
                               col_B: str,
                               order: int=1) -> float:
    num_list_A = df[col_A]
    num_list_B = df[col_B]
    
    if any(val != 0 for val in num_list_A):
        num_list_A = num_list_A / sum(num_list_A)
    
    if any(val != 0 for val in num_list_B):
        num_list_B = num_list_B / sum(num_list_B)
    
    return np.linalg.norm(num_list_A - num_list_B, order)


def get_partitions(min_dt: datetime.datetime,
                   max_dt: datetime.datetime,
                   prtitn_nbr: int=1,
                   min_days: int=30,
                   random_state: int=2024) -> List[datetime.datetime]:
    if max_dt <  min_dt + datetime.timedelta(min_days * (prtitn_nbr+1)):
        raise ValueError('Not enough days for partitions')
    
    remain_days = (max_dt - min_dt).days - min_days * (prtitn_nbr+1)
    
    random.seed(random_state)
    random_int_list = sorted([random.randint(0, remain_days) for _ in range(prtitn_nbr)])
    
    return [min_dt + datetime.timedelta(days = min_days*i + days) \
                for i, days in enumerate(random_int_list, 1)]


def repartitions_using_distance(df: pd.DataFrame,
                                dt_list: List[datetime.datetime],
                                fit_model_func,
                                get_distance_func,
                                min_thrhld: float,
                                max_thrhld: float) -> List[datetime.datetime]:
    dt_list.insert(0, df.index.min())
    dt_list.insert(len(dt_list), df.index.max())
    dt_list = sorted(list(set(dt_list)))
    dt_list = [dt for dt in dt_list if (df.index.min() <= dt) and (dt <= df.index.max())]
    
    prev_df = df[(df.index >= dt_list[0]) & (df.index <= dt_list[1])]
    prev_model = fit_model_func(prev_df)
    i = 1
    while i < len(dt_list)-1:
        curr_df = df[(df.index >= dt_list[i]) & (df.index <= dt_list[i+1])]
        curr_model = fit_model_func(curr_df)
        if get_distance_func(prev_model, curr_model) < min_thrhld:
            print(f'Removing {dt_list[i]}')
            del dt_list[i]
            if i < len(dt_list)-2:
                prev_df = df[(df.index >= dt_list[i]) & (df.index <= dt_list[i+1])]
                prev_model = fit_model_func(prev_df)
        prev_model = curr_model
        i += 1
    
    prev_df = df[(df.index >= dt_list[0]) & (df.index <= dt_list[1])]
    prev_model = fit_model_func(prev_df)
    i = 1
    while i < len(dt_list)-1:
        curr_df = df[(df.index >= dt_list[i]) & (df.index <= dt_list[i+1])]
        curr_model = fit_model_func(curr_df)
        if get_distance_func(prev_model, curr_model) > max_thrhld:
            next_dt = dt_list[i+1]
            if dt_list[i+1] >=  dt_list[i-1] + datetime.timedelta(30*(2+1)):
                print(f'Adding at {dt_list[i]}')
                del dt_list[i]
                dt_list[i:i] = get_partitions(min_dt=dt_list[i-1],
                                              max_dt=dt_list[i+1],
                                              prtitn_nbr=2,
                                              min_days=30)
                i = dt_list.index(next_dt)
                if i < len(dt_list)-2:
                    prev_df = df[(df.index >= dt_list[i]) & (df.index <= dt_list[i+1])]
                    prev_model = fit_model_func(prev_df)
        else:
            prev_model = curr_model
        i += 1
    print(f'----------------- length: {len(dt_list)} -----------------')
    return dt_list
