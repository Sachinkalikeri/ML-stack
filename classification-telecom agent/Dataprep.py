# Remove unnecessary columns
columns_to_drop = ['column1', 'column2', ...]  # Specify the columns you want to drop
df1 = df1.drop(columns_to_drop, axis=1)

# Handle missing values
df1 = df1.fillna(method='ffill')  # Forward fill missing values with the previous valid value

# Convert categorical variables to numerical
categorical_columns = ['column3', 'column4', ...]  # Specify the categorical columns
df1 = pd.get_dummies(df1, columns=categorical_columns)

# Normalize numerical columns
numerical_columns = ['column5', 'column6', ...]  # Specify the numerical columns
scaler = StandardScaler()
df1[numerical_columns] = scaler.fit_transform(df1[numerical_columns])

# Split the data into features and target variable
X = df1.drop('target_column', axis=1)
y = df1['target_column']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

