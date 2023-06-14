# Importing Libraries
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# Reading the dataset
df = pd.read_csv('/kaggle/input/coupons/Coupons.csv')

# Substituting numerical values for each category in 'Coupon' variable
df['Coupon'] = df['Coupon'].map({'Kids Apparel': 0, 'Womens Apparel': 1, 'Mens Apparel': 2}).astype(int)

# Substituting numerical values for each category in 'Gender' variable
df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1}).astype(int)

# Substituting numerical values for each category in 'Item1' variable
df['Item1'] = df['Item1'].map({'Computer Games': 0, 'Mens Wear': 1, 'Cosmetics': 2, 'Heels': 3, 'Handbag': 4}).astype(int)

# Substituting numerical values for each category in 'Item2' variable
df['Item2'] = df['Item2'].map({'Movies': 0, 'Womens Wear': 1, 'Toys': 2, 'Board Games': 3, 'Gym Shoes': 1}).astype(int)

# Substituting numerical values for each category in 'Item3' variable
df['Item3'] = df['Item3'].map({'Educational Products': 0, 'Tie': 1, 'Kids Wear': 2, 'Candy': 2}).astype(int)

# Selecting predictor and target variables
X = df.iloc[:, 0:5]
y = df.iloc[:, 5]

# Train and Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# Standardizing the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
