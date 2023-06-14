import numpy as np
import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('../input/employee-satisfaction-index-dataset/Employee Satisfaction Index.csv')

# Remove the 'Unnamed: 0' column
df = df.drop(['Unnamed: 0'], axis=1)

# Convert text data to numeric representations
df = df.replace(['HR', 'Technology', 'Sales', 'Purchasing', 'Marketing'], [1, 2, 3, 4, 5])
df = df.replace(['Suburb', 'City'], [1, 2])
df = df.replace(['PG', 'UG'], [1, 2])
df = df.replace(['Referral', 'Walk-in', 'On-Campus', 'Recruitment Agency'], [1, 2, 3, 4])

# Display the head of the DataFrame
df.head()

