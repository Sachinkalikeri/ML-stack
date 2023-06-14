from sklearn.preprocessing import StandardScaler

# Remove columns with more than 50 unique values
df1 = df1.loc[:, df1.nunique() <= 50]

# Drop columns with NaN values
df1 = df1.dropna(axis='columns')

# Standardize the numerical columns
numerical_cols = df1.select_dtypes(include=np.number).columns
scaler = StandardScaler()
df1[numerical_cols] = scaler.fit_transform(df1[numerical_cols])

