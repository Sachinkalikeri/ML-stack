import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Loading and inspection - Carga e Inspección
dataset = pd.read_csv('../input/call-center-metrics-dataset/call_metrics_dataset.csv', delimiter=';')
dataset.index = pd.to_datetime(dataset['date'])
del(dataset['date'])
dataset['avg_aht'] = dataset['avg_aht'].str.replace('.', '', regex=True)
dataset['avg_aht'] = dataset['avg_aht'].str.replace(',', '.', regex=True).astype(float)

# Splitting the dataset - Dividiendo el dataset
x = dataset.iloc[:, 0:6].values
y = dataset.iloc[:, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Normalizing the variables - Normalizando las variables
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Using variable reduction (PCA) - Usando reducción de variables (PCA)
pca = PCA(n_components=3)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_

