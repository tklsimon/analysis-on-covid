import pandas as pd

from statsmodels.tsa.stattools import adfuller
from typing import List


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


# def get_date_count(df: pd.DataFrame,
#                    col: str,
#                    date_format: str) -> pd.DataFrame:
#     df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
#     agg_df = df.groupby(col)[col].count()
    
#     date_idx = pd.date_range(agg_df.index.min(), agg_df.index.max())
#     agg_series = pd.Series(agg_df)
#     agg_series.index = pd.DatetimeIndex(agg_series.index)
#     agg_series = agg_series.reindex(date_idx, fill_value=0)
    
#     return pd.DataFrame({col: agg_series.index, 'count': agg_series.values})


def add_lag_columns(df: pd.DataFrame,
                    col: str,
                    lag_nbr: int) -> None:
    for nbr in range(1, lag_nbr+1):
        df[f'{col}_lag{nbr}'] = df[col].shift(nbr).fillna(0)


def stationary_and_difference(df: pd.DataFrame,
                              col_list: List[str]=None) -> None:
    if col_list is None:
        col_list = df.columns
    for col in col_list:
        p_value = adfuller(df[col])[1]
        if p_value > 0.05:
            print(f'The column {col} has ADF p-value {p_value:.5f} which is non-stationary')
            
            new_diff_col = col + '_diff_1'
            print(f'Creating a difference column {new_diff_col} ...')
            df[new_diff_col] = df[col].diff().fillna(0)
            print(f'Droping the original column {col} ...')
            df.drop(col, axis=1, inplace=True)
        else:
            print(f'The column {col} has ADF p-value {p_value:.5f} which is stationary.')
