import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Dataset
df = pd.read_csv('/kaggle/input/car-acceptability-classification-dataset/car.csv')

# Converting Categorical variables into numeric values
LE = LabelEncoder()
df['Buying_Price'] = LE.fit_transform(df['Buying_Price'])
df['Maintenance_Price'] = LE.fit_transform(df['Maintenance_Price'])
df['Size_of_Luggage'] = LE.fit_transform(df['Size_of_Luggage'])
df['Safety'] = LE.fit_transform(df['Safety'])
df['Car_Acceptability'] = LE.fit_transform(df['Car_Acceptability'])
df['No_of_Doors'] = LE.fit_transform(df['No_of_Doors'])
df['Person_Capacity'] = LE.fit_transform(df['Person_Capacity'])

