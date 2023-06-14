import numpy as np
import pandas as pd
from sklearn import preprocessing

# Reading the dataset
df = pd.read_csv('../input/job-classification-dataset/jobclassinfo2.csv')

# Checking the information about the dataset
df.info()

# Displaying the first 3 rows of the dataset
df.head(3)

# Checking for missing values
df.isnull().sum()

# Selecting the object type features
object_type_features = df.select_dtypes("object").columns

# Label encoding the object type features
le = preprocessing.LabelEncoder()
for feat_name in object_type_features:
    df[feat_name] = le.fit_transform(df[feat_name])

# Checking the updated information about the dataset
df.info()

# Displaying the updated dataset
df.head()

# Separating the features (X) and the target variable (y)
X = df.drop(['PG'], axis=1)
y = df['PG']

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

