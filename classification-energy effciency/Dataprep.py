import pandas as pd  
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

# Load the dataset
energy_df = pd.read_csv("/kaggle/input/energy-efficiency-data-set/energy_efficiency_data.csv")

# View information about the data types and missing values
energy_df.info()
energy_df.shape

# View the first five rows of the data
energy_df.head()

# View the last five rows of the data
energy_df.tail()

# View summary statistics of the data
energy_df.describe()

