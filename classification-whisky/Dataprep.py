import pandas as pd

# Read the whisky dataset
df = pd.read_csv('/kaggle/input/scotch-whisky-dataset/whisky.csv', index_col=0)

# Check for missing values
df.isna().sum()

