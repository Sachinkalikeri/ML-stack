import pandas as pd

# Read the CSV file
df = pd.read_csv('/kaggle/input/indian-premier-league-ipl-all-seasons/all_season_summary.csv')

# Check the information about the DataFrame
df.info()

# Check the shape of the DataFrame
print("Shape of the DataFrame:", df.shape)

# Check for missing values
print("Missing values in each column:")
print(df.isna().sum())

# Select specific columns
selected_columns = ['home_score', 'short_name', 'winner']
subset_df = df[selected_columns].head(20)
print(subset_df)

# Drop columns
columns_to_drop = ['column1', 'column2', 'column3']
df = df.drop(columns_to_drop, axis=1)

# Fill missing values
df['column_name'] = df['column_name'].fillna(value)

# Convert data types
df['column_name'] = df['column_name'].astype(new_data_type)

# Perform other necessary preprocessing steps such as encoding categorical variables, scaling numeric data, etc.

