# First, handle the categorical columns
le = LabelEncoder()

for col in ['sex','cp','restecg','slp','thall']:
    df[col] = le.fit_transform(df[col])

# Scale numerical features
scaler = StandardScaler()

for col in ['age','trtbps','chol','fbs','thalachh','exng','oldpeak','caa']:
    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
    2. t-SNE
# Compute t-SNE
X = df.drop('output', axis=1).values  # dropping the target column
y = df['output'].values  # the target column

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

# Create a new DataFrame for the two-dimensional t-SNE representation
df_tsne = pd.DataFrame(data = X_tsne, columns = ['Component 1', 'Component 2'])
df_tsne['Target'] = y

# Visualize with Plotly
fig = px.scatter(df_tsne, x='Component 1', y='Component 2', color='Target', 
                 title='2 Component t-SNE', template='plotly')
fig.show()
