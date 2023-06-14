# Check for duplicates
print("\nNumber of duplicate rows:", data.duplicated().sum())
# Handle duplicates
data.drop_duplicates(inplace=True)
# Updated dataset after handling missing values and duplicates
print("\nDataset shape after handling missing values and duplicates:", data.shape)
data.dtypes
