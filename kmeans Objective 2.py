import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

# Load your data
df = pd.read_csv('retractions35215.csv')

# Encode categorical data properly
for column in df.select_dtypes(include=[object]).columns:
    df[column] = LabelEncoder().fit_transform(df[column].astype(str))

# Handling NaNs and Infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)  # Adjust according to your data analysis

# Assume 'Journal' and 'Publisher' are two predictors
features = df.columns.drop('Reason')

# Generating Polynomial and Interaction features
# For example, using degree 2 polynomial features including interactions
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(df[['RetractionDOI', 'OriginalPaperDOI']])  # Focus on DOI features for clustering
poly_features = poly.get_feature_names_out(['RetractionDOI', 'OriginalPaperDOI'])
X_poly_df = pd.DataFrame(X_poly, columns=poly_features)

# K-Means Clustering model with 3 clusters
kmeans_param = 3
model = KMeans(n_clusters=kmeans_param, max_iter=100, random_state=0)
clusters = model.fit_predict(X_poly_df)

# Visualise clustering results
plt.figure(figsize=(12, 6))
plt.scatter(X_poly_df['RetractionDOI OriginalPaperDOI'], X_poly_df['RetractionDOI^2'], c=clusters, cmap="jet")
plt.title("K-Means Clustering on DOI Features", fontsize=18)
plt.xlabel("RetractionPaperDOI and OriginalPaperDOI Interaction")
plt.ylabel("Square of RetractionPaperDOI")
plt.colorbar(label='Cluster Label')
plt.show()
