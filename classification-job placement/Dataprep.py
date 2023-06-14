import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, KBinsDiscretizer
from sklearn.impute import KNNImputer

job_placement = pd.read_csv("/kaggle/input/job-placement-dataset/Job_Placement_Data.csv")
records = pd.DataFrame(data=job_placement)

numeric_columns = records.select_dtypes(include=['float']).columns
scaler = MinMaxScaler()
records[numeric_columns] = scaler.fit_transform(records[numeric_columns])

scaler = StandardScaler()
records[numeric_columns] = scaler.fit_transform(records[numeric_columns])

discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform")
discretized_values = discretizer.fit_transform(records[numeric_columns])
discretized_df = pd.DataFrame(discretized_values, columns=numeric_columns)

knn_imputer = KNNImputer(n_neighbors=3)
records.loc[:, numeric_columns] = knn_imputer.fit_transform(records.loc[:, numeric_columns])

