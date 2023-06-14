import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Data
df = pd.read_csv("/kaggle/input/name-data-set/Indian-Name-12.csv", encoding='unicode_escape')

# Modify Dataframe / Add New Columns
df["Name"] = df["Name"].str.lower()
df["name_length"] = df["Name"].str.len()
alphabet = "abcdefghijklmnopqrstuvwxyz"
for letter in alphabet:
    df[f"begins_with_{letter}"] = df["Name"].apply(lambda n: 1 if n[0] == letter else 0)
    df[f"ends_with_{letter}"] = df["Name"].apply(lambda n: 1 if n[-1] == letter else 0)
    df[f"count_{letter}"] = df["Name"].apply(lambda n: n.count(letter))

# Create Feature Matrices
X = df.drop(["Name", "Target"], axis=1).to_numpy()
y = df["Target"].to_numpy()

# Split Test and Training Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Scale / Normalize Data
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

