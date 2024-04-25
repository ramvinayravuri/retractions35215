import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from matplotlib import pyplot as plt

# Load your data
df = pd.read_csv('retractions35215.csv')

# Encode categorical data
for col in df.select_dtypes(include=['object']):
    df[col] = LabelEncoder().fit_transform(df[col])

# Fill missing values
df.fillna(df.mean(), inplace=True)

# Define response variable and features
X = df.drop('Reason', axis=1)  # Ensure 'CitationCount' exists in your DataFrame
y = df['Reason']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.6, random_state=0)

# Define the model and feature selector
model = LinearRegression()
sfs = SFS(model, k_features=5, forward=True, scoring='neg_mean_squared_error', cv=5)

# Fit SFS
sfs = sfs.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[list(sfs.k_feature_idx_)]
print("Selected features:", selected_features)

# Fit the final model
model.fit(X_train[selected_features], y_train)

# Print model summary (using statsmodels for summary output)
import statsmodels.api as sm
X_sm = sm.add_constant(X_train[selected_features])  # Adds a constant term to the predictor
est = sm.OLS(y_train, X_sm)
est2 = est.fit()
print(est2.summary())

# Sample data: Replace these lists with your actual R² values and standard errors
number_of_features = np.arange(1, 13)  # Number of features
performance = np.random.rand(12) * 0.2 + 0.55  # Replace with actual R² values
standard_errors = np.random.rand(12) * 0.05  # Replace with actual StdErr values

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(number_of_features, performance, marker='o', color='b', label='R² Score')
plt.fill_between(number_of_features, performance - standard_errors, performance + standard_errors, color='b', alpha=0.2)

# Labeling the plot
plt.title('Sequential Forward Selection (w. StdErr)')
plt.xlabel('Number of Features')
plt.ylabel('Performance (R²)')
plt.grid(True)

# Show plot
plt.show()