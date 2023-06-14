import numpy as np
import pandas as pd

# Importing the dataset and checking its structure
data = pd.read_csv('../input/world-happiness/2019.csv')
print(data.info())
data.head(10)
data.describe()

# Renaming the 'Country or region' column to 'Country'
data.rename(columns={'Country or region':'Country'}, inplace=True)
data.info()

# Removing missing values from the dataset
if data.isnull().values.any():
    data.dropna(subset=[
        'Overall rank',
        'Country',
        'Score',
        'GDP per capita',
        'Social support',
        'Healthy life expectancy',
        'Freedom to make life choices',
        'Generosity',
        'Perceptions of corruption'
    ], inplace=True)
else:
    print('No missing values found')


