from typing import List

import numpy as np
import pandas as pd
from hmmlearn import hmm

## Example prediction
data_sequence: List = [1, 2, 3, 2, 1, 2, 3, 4, 3, 2]
sample_sequence = np.array(data_sequence).reshape(-1, 1)

sample_model = hmm.MultinomialHMM(n_components=2)
sample_model.fit(sample_sequence)

print("Transition matrix:")
print(sample_model.transmat_)
print("Emission matrix:")
print(sample_model.emissionprob_)

# model evaluation
print(sample_model.score(sample_sequence))
print(sample_model.aic(sample_sequence))
print(sample_model.bic(sample_sequence))

print("==================")


### Import Dataset
df_covid = pd.read_csv("data/std_data/hk/covid_hk_std.csv")
df_covid['report_date'] = pd.to_datetime(df_covid['report_date'], dayfirst=True) # convert to datetime type
print(df_covid.shape)


def get_date_count(df: pd.DataFrame, col: str) -> pd.DataFrame:
    agg_df = df.groupby(col)[col].count()
    date_idx = pd.date_range(agg_df.index.min(), agg_df.index.max())
    agg_series = pd.Series(agg_df)
    agg_series.index = pd.DatetimeIndex(agg_series.index)
    agg_series = agg_series.reindex(date_idx, fill_value=0)
    return pd.DataFrame({col: agg_series.index, 'count': agg_series.values})


df_count = get_date_count(df_covid, 'report_date')
print(df_count.head())
print(df_count.shape)


sequence = df_count['count'].to_numpy().reshape(-1, 1)
wave_4 = sequence[350:400] # forth wave

hmm_model = hmm.MultinomialHMM(n_components=2, verbose=False, n_iter=1000, tol=1e-3)
hmm_model.fit(wave_4)

print(hmm_model.transmat_)

# model evaluation
print(hmm_model.score(wave_4))
print(hmm_model.aic(wave_4))
print(hmm_model.bic(wave_4))