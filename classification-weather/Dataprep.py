# Handling missing values
df1 = df1.dropna()  # Drop rows with any missing values

# Encoding categorical variables (if any)
# Assuming there is a categorical column named 'category'
# You can use one-hot encoding or label encoding depending on the context
# One-hot encoding example:
df1 = pd.get_dummies(df1, columns=['category'])

# Scaling numerical features
# Assuming there are numerical columns that need to be scaled
# You can use StandardScaler from sklearn.preprocessing
scaler = StandardScaler()
numerical_columns = ['column1', 'column2', 'column3']  # Replace with actual column names
df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])
