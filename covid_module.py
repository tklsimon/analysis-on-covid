import pandas as pd

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
    df_na_cnt = df.isnull().sum()
    df_record_cnt = df.shape[0]
    
    # Print the count of missing value for each feature
    if df_na_cnt.sum() > 0:
        print('The following columns have missing values:')
        for col, na_cnt in zip(df_na_cnt.index, df_na_cnt.values):
            if na_cnt > 0:
                print(f'{col}: {na_cnt} ({100*na_cnt/df_record_cnt:0.1f}%)')
    else:
        print('This dataframe does not have missing values.')


def change_date_format(df: pd.DataFrame,
                       old_column_name: str,
                       old_date_format: str,
                       new_column_name: str,
                       new_date_format: str) -> None:
    df[new_column_name] = \
        pd.to_datetime(df[old_column_name], format=old_date_format, errors='coerce') \
        .dt.strftime(new_date_format) \
        .fillna('N/A')


def get_year_month_part(df: pd.DataFrame,
                        column_name: str,
                        date_format: str) -> None:
    for (date_part, date_part_format) in [('year','%Y'), ('year_month','%Y%m')]:
        new_column_name = column_name.removesuffix('_date') + '_' + date_part
        change_date_format(df,
                           column_name,
                           date_format,
                           new_column_name,
                           date_part_format)


def one_hot_encoding(df: pd.DataFrame,
                     column_name: str) -> pd.DataFrame:
    distinct_values = set(df[column_name])
    
    for value in distinct_values:
        new_column_name = column_name + '_' + value
        df[new_column_name] = df[column_name].apply(lambda col: 1 if col == value else 0)
    
    return df


def get_date_count(df: pd.DataFrame,
                   col: str,
                   date_format: str) -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], format=date_format, errors='coerce')
    agg_df = df.groupby(col)[col].count()
    
    date_idx = pd.date_range(agg_df.index.min(), agg_df.index.max())
    agg_series = pd.Series(agg_df)
    agg_series.index = pd.DatetimeIndex(agg_series.index)
    agg_series = agg_series.reindex(date_idx, fill_value=0)
    
    return pd.DataFrame({col: agg_series.index, 'count': agg_series.values})