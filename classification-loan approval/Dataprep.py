import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Reading the dataset
data = pd.read_csv('/kaggle/input/loan-approval-systemlas/clientes.csv')
data.head()

# Dropping unnecessary columns
data.drop(columns=['cod_cliente'], inplace=True)

# Handling missing values
data.sexo = data.sexo.fillna('No')
data.estado_civil = data.estado_civil.fillna('No')
data.empregado = data.empregado.fillna('No')
data.dependentes = data.dependentes.fillna(0)
data.emprestimo = data.emprestimo.fillna(data.emprestimo.mean())
data.prestacao_mensal = data.prestacao_mensal.fillna(data.prestacao_mensal.mean())
data.historico_credito = data.historico_credito.fillna(0.0)

# Encoding categorical variables
data["dependentes"].replace({"3+": "3"}, inplace=True)
data["renda_conjuge"].replace({"9.857.999.878": "9875", "1.612.000.084": "1612"}, inplace=True)

# Splitting the data into features and target
x = data.drop(columns=['aprovacao_emprestimo'])
y = data[['aprovacao_emprestimo']]

# Label Encoding the target variable
le = LabelEncoder()
y[['aprovacao_emprestimo']] = pd.DataFrame(le.fit_transform(y[['aprovacao_emprestimo']]))

# Splitting the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

