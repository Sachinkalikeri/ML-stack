import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv('/kaggle/input/easiest-diabetes-classification-dataset/Diabetes Classification.csv')

# Encoding categorical variables
BloodOrder = ['Low', 'Normal', 'High']
FamHis = ['No', 'Yes']
SmoHis = ['No', 'Yes']
DietOr = ['Poor', 'Healthy']
ExcerOr = ['No', 'Regular']

label_encoder = LabelEncoder()
label_encoder.fit(BloodOrder)
df['Blood Pressure'] = label_encoder.transform(df['Blood Pressure'])

label_encoder.fit(FamHis)
df['Family History of Diabetes'] = label_encoder.transform(df['Family History of Diabetes'])

label_encoder.fit(SmoHis)
df['Smoking'] = label_encoder.transform(df['Smoking'])

label_encoder.fit(DietOr)
df['Diet'] = label_encoder.transform(df['Diet'])

df['Exercise'].replace({'No': 0, 'Regular': 1}, inplace=True)
df['Diagnosis'].replace({'No': 0, 'Yes': 1}, inplace=True)
encoded_features = pd.get_dummies(df['Gender'], prefix='Gender')
df = pd.concat([df.drop('Gender', axis=1), encoded_features], axis=1)

# Splitting the data into train and test sets
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Scaling the numerical features
valcol = df.select_dtypes(exclude=['object']).columns.tolist()
scaler = StandardScaler()
scaler.fit(X_train[valcol])
X_train_scaled = X_train.copy()
X_train_scaled[valcol] = scaler.transform(X_train[valcol])
X_test_scaled = X_test.copy()
X_test_scaled[valcol] = scaler.transform(X_test[valcol])

# Fitting the SVM model
svm_model = SVC()
svm_model.fit(X_train_scaled, y_train)
y_pred_train = svm_model.predict(X_train_scaled)
y_pred_test = svm_model.predict(X_test_scaled)

# Calculating accuracy scores
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

# Plotting confusion matrix
ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Labels")
plt.ylabel("Actual Labels")
plt.title("Confusion Matrix")
plt.show()

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

