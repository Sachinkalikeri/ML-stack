import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
metro_data = pd.read_csv('/kaggle/input/-world-metro/metro_countries_total.csv')

# Check for missing data points
missing_values_count = metro_data.isnull().sum()
print(missing_values_count)

# Perform any necessary preprocessing steps here, such as handling missing values, encoding categorical variables, etc.
# ...

# Example: Standardize numerical features using StandardScaler
scaler = StandardScaler()
numerical_features = ['length', 'stations', 'annual_ridership_mill']
metro_data[numerical_features] = scaler.fit_transform(metro_data[numerical_features])

# Continue with further preprocessing if needed
# ...

