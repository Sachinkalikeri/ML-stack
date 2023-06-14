
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Reading the dataset
df = pd.read_csv("/kaggle/input/milkquality/milknew.csv")
display(df.head(2))
df.info()

# Checking unique values in each column
for i in df.columns:
    print(f"column {i}")
    print(df[i].unique())
    print(f"unique number of elements in column {i} is {df[i].nunique()}")
    print("")

# Encoding the 'Grade' column
df.Grade = df.Grade.replace({"high": 3, "medium": 2, "low": 1}).astype("int")
df.head(2)

# Preprocessing for modeling
scaler = StandardScaler()
x = df.iloc[:, 0:8]
y = df.iloc[:, -1]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, train_size=0.85)
