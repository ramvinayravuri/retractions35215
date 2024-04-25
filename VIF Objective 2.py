import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load your data
df = pd.read_csv('retractions35215.csv')

# Encode categorical data and convert non-numeric data to numeric, handling errors
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = LabelEncoder().fit_transform(df[column].astype(str))
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Handle missing values with mean imputation (consider other methods depending on your analysis)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)

# Define the dependent variable and features
response_var = 'CitationCount'  # adjust this to your actual dependent variable
X = df.drop(columns=[response_var])
y = df[response_var]

# Generating Polynomial and Interaction features for a subset of predictors
# Assuming we select specific predictors for demonstration
selected_columns = ['Journal', 'Publisher']  # adjust these columns based on your dataset
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X[selected_columns])
poly_features = poly.get_feature_names_out(selected_columns)
X_poly_df = pd.DataFrame(X_poly, columns=poly_features)

# Combine polynomial features with original data, excluding original columns used for polynomial transformation
X = pd.concat([X.drop(selected_columns, axis=1), X_poly_df], axis=1)

# Add a constant to the independent variables matrix
X = sm.add_constant(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the ordinary least squares regression model
model = sm.OLS(y_train, X_train).fit()
print("Initial Model Summary:")
print(model.summary())

# Backward Elimination function with p-value threshold
def backward_elimination(X, y, significance_level=0.05):
    features = X.columns.tolist()
    while len(features) > 0:
        current_model = sm.OLS(y, X[features]).fit()
        p_values = current_model.pvalues[1:]  # Exclude intercept
        max_p = max(p_values, default=0)
        if max_p > significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
            print(f"Dropping {excluded_feature} with p-value {max_p}")
        else:
            break
    return current_model

# Applying the backward elimination
final_model = backward_elimination(X_train, y_train)

# Print the summary of the final model
print("Final Model Summary:")
print(final_model.summary())

# Check for multicollinearity among the remaining features
vif_data = pd.DataFrame()
vif_data['feature'] = final_model.params.index
vif_data['VIF'] = [variance_inflation_factor(X_train[final_model.params.index].values, i) for i in range(X_train[final_model.params.index].shape[1])]

print("VIF Scores:")
print(vif_data)

# Assuming 'X_train' is your training dataset from previous steps
reduced_X = X_train.drop(columns=['Journal', 'Publisher', 'Journal^2', 'Publisher^2'])

# Recompute the model without these features
reduced_model = sm.OLS(y_train, reduced_X).fit()
print(reduced_model.summary())

# Recalculate VIF scores
new_vif_data = pd.DataFrame()
new_vif_data['feature'] = reduced_X.columns
new_vif_data['VIF'] = [variance_inflation_factor(reduced_X.values, i) for i in range(reduced_X.shape[1])]

print("New VIF Scores:")
print(new_vif_data)

# Remove high VIF DOI variables for reanalysis
reduced_X = reduced_X.drop(columns=['RetractionDOI', 'OriginalPaperDOI'])

# Remove the 'OriginalPaperPubMedID' to reduce multicollinearity
reduced_X = reduced_X.drop(columns=['OriginalPaperPubMedID'])

# Refit the model without 'OriginalPaperPubMedID'
refitted_model = sm.OLS(y_train, reduced_X).fit()
print(refitted_model.summary())

# Recalculate VIF scores to see the impact of removal
refitted_vif_data = pd.DataFrame()
refitted_vif_data['feature'] = reduced_X.columns
refitted_vif_data['VIF'] = [variance_inflation_factor(reduced_X.values, i) for i in range(reduced_X.shape[1])]

print("Revised VIF Scores After Removal:")
print(refitted_vif_data)
