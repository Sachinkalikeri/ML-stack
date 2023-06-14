import pandas as pd
from sklearn.preprocessing import FunctionTransformer, PowerTransformer, StandardScaler

# Read the dataset
df = pd.read_csv("/kaggle/input/eligibility-prediction-for-loan/Loan_Data.csv")

# Drop Loan_ID column
df.drop(columns=['Loan_ID'], inplace=True)

# Fill missing values
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)

# Log transformation
Func = FunctionTransformer(func=np.log1p)
df['ApplicantIncome'] = Func.fit_transform(df['ApplicantIncome'])

# Power transformation
pt = PowerTransformer(method='yeo-johnson', standardize=False)
df[['Loan_Amount_Term', 'LoanAmount', 'CoapplicantIncome']] = pt.fit_transform(df[['Loan_Amount_Term', 'LoanAmount', 'CoapplicantIncome']])

# Standard scaling
scaling = StandardScaler()
df[['Loan_Amount_Term', 'LoanAmount', 'CoapplicantIncome', 'ApplicantIncome']] = scaling.fit_transform(df[['Loan_Amount_Term', 'LoanAmount', 'CoapplicantIncome', 'ApplicantIncome']])

# One-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

