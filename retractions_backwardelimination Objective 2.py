import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures

# Load your data
df = pd.read_csv('retractions35215.csv')

# Assuming 'CitationCount' or another specific column as the response variable
response_var = 'CitationCount'
features = df.columns.drop(response_var)

# Encode categorical data properly
for column in df.select_dtypes(include=[object]).columns:
    if column in features:
        df[column] = LabelEncoder().fit_transform(df[column].astype(str))

# Handling NaNs and Infinities
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(df.mean(), inplace=True)  # Adjust according to your data analysis

# Selecting a few predictors for polynomial and interaction terms
# Assume 'Journal' and 'Publisher' are two such predictors
X = df[features]
y = df[response_var]

# Generating Polynomial and Interaction features
# For example, using degree 2 polynomial features including interactions
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X[['Record ID', 'Title', 'Subject', 'Institution', 'Country', 'Author', 'URLS', 'ArticleType', 'RetractionDate', 'RetractionPubMedID', 'OriginalPaperDate', 'Paywalled', 'Notes','Journal', 'Publisher']])  # adjust columns as needed
poly_features = poly.get_feature_names_out(['Record ID', 'Title', 'Subject', 'Institution', 'Country', 'Author', 'URLS', 'ArticleType', 'RetractionDate', 'RetractionPubMedID', 'OriginalPaperDate', 'Paywalled', 'Notes','Journal', 'Publisher'])
X_poly_df = pd.DataFrame(X_poly, columns=poly_features)

# Combine with original data
X = pd.concat([X.drop(['Record ID', 'Title', 'Subject', 'Institution', 'Country', 'Author', 'URLS', 'ArticleType', 'RetractionDate', 'RetractionPubMedID', 'OriginalPaperDate', 'Paywalled', 'Notes','Journal', 'Publisher'], axis=1), X_poly_df], axis=1)

# Add a constant to the model (for the intercept)
X = sm.add_constant(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# Building and fitting the initial model
model = sm.OLS(y_train, X_train).fit()
# print("Model Summary with Polynomial and Interaction Features:")
# print(model.summary())
# Backward Elimination function with R-squared consideration
def backward_elimination(X, y, significance_level=0.05, r_squared_threshold=0.7):
    features = X.columns.tolist()
    current_model = sm.OLS(y, X).fit()
    while len(features) > 1 and current_model.rsquared > r_squared_threshold:
        p_values = current_model.pvalues[1:]  # Exclude intercept
        max_p = max(p_values)
        if max_p > significance_level:
            excluded_feature = p_values.idxmax()
            features.remove(excluded_feature)
            X = X[features]
            current_model = sm.OLS(y, X).fit()
            print(f"Dropping {excluded_feature} with p-value {max_p}")
        else:
            break
    return current_model

# Applying the backward elimination
final_model = backward_elimination(X_train, y_train, r_squared_threshold=0.7)

# Print the summary of the final model
print("Final Model Summary:")
print(final_model.summary())

# Listing top 5 most significant variables by their coefficients
top_vars = final_model.params[1:].abs().sort_values(ascending=False).head(5)
print("Top 5 Significant Variables:")
print(top_vars)
