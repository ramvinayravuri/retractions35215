import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt


# Load the data
df = pd.read_csv('retractions.csv')

# Encode categorical data
label_encoder = LabelEncoder()
for column in df.select_dtypes(include=[object]).columns:
    df[column] = label_encoder.fit_transform(df[column].astype(str))

# Handle missing values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)

# Separate the features and the response variable
response_var = 'CitationCount'
features = df.columns.drop(response_var)

X = df[features]
y = df[response_var]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.4, random_state=0)

# Apply PCA
pca = PCA(n_components=15) 

# Fit PCA on the training data and transform both training and testing data
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
print(abs(pca.components_)) # To see contribution of each explanatory variable in each principal component

# Examine explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:")
# print(explained_variance_ratio)

# Store feature names before PCA
feature_names = list(X.columns)

# Apply PCA
pca = PCA(n_components=None)  

# Fit PCA on the training data and transform both training and testing data
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Plot the cumulative explained variance versus number of PCA components
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xticks(range(1,20))
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.grid()
plt.show()

# Calculate covariance matrix for deeper analysis
covariance_matrix = np.cov(X_scaled, rowvar=False) 
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sort components by their importance based on eigenvalues
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Enhanced component selection analysis (consider Kaiser-Guttman criterion)
print(eigenvalues)

# Map feature names onto PCA components
for i, component in enumerate(pca.components_):
    print(f"Principal Component {i+1}:")
    contributions = pd.DataFrame({'Feature Name': feature_names, 
                                  'Contribution': abs(component)})
    contributions.sort_values(by='Contribution', ascending=False, inplace=True)
    print(contributions)

