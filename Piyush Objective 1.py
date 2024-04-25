import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Displaying all columns of the dataframe
pd.pandas.set_option("display.max_columns",None)
df = pd.read_csv("D:\\PRT564 DATA ANALYTICS AND VISUALISATION\\Group Project\\Objective 1\\retractions35215.csv")


# Get the count of missing values in each column
missing_values_count = df.isnull().sum()

# Calculate the percentage of missing values in each column
total_values = df.shape[0]
missing_values_percentage = (missing_values_count / total_values) * 100


# Combine the count and percentage into a DataFrame for display
missing_data_summary = pd.DataFrame({'Missing Values Count': missing_values_count,
                                     'Missing Values Percentage': missing_values_percentage})

# Display the count and percentage of missing values for each column
print("Summary of Missing Values in Each Column:")
print(missing_data_summary)

# Convert 'OriginalPaperDate' and 'RetractionDate' columns to datetime objects
df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], format='%d/%m/%Y')
df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], format='%d/%m/%Y')

# Calculate the time difference between 'OriginalPaperDate' and 'RetractionDate' in months
df['time_to_retraction_months'] = (df['RetractionDate'] - df['OriginalPaperDate']).dt.days / 30

# Display the newly calculated column
print(df[['OriginalPaperDate', 'RetractionDate', 'time_to_retraction_months']].head())

# Get the features with missing values
features_with_missing_values = df.columns[df.isnull().any()]

# Plot countplots for features with missing values
for feature in features_with_missing_values:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=df[feature].isnull(), palette='binary')
    plt.title(f'Missing Values in {feature}')
    plt.xlabel('Missing Values (1: Missing, 0: Not Missing)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Not Missing', 'Missing'])
    plt.tight_layout()
    plt.show()


# Plot the effect of missing values in each column on retraction time using box plots
for feature in features_with_missing_values:
    plt.figure(figsize=(8, 6))
    # Convert boolean values to integers
    x_values = df[feature].isnull().astype(int)
    sns.barplot(x=x_values, y=df['time_to_retraction_months'], palette='Set2')
    plt.title(f'Effect of Missing Values in {feature} on Retraction Time (in Months)')
    plt.xlabel('Missing Values (1: Missing, 0: Not Missing)')
    plt.ylabel('Time to Retraction (Months)')
    plt.xticks([0, 1], ['Not Missing', 'Missing'])
    plt.tight_layout()
    plt.show()

# Get the count of numerical features in the dataset
numerical_features_count = df.select_dtypes(include=np.number).shape[1]

# Display the number of numerical features
print("Number of Numerical Features in the Dataset:", numerical_features_count)

# Manually enter the features with year information
features_with_year = ['RetractionDate','OriginalPaperDate']

# Plot the features with year information with respect to retraction time
for feature in features_with_year:
    plt.figure(figsize=(8, 6))
    plt.scatter(df[feature], df['time_to_retraction_months'], alpha=0.5)
    plt.title(f'{feature} vs Time to Retraction')
    plt.xlabel(feature)
    plt.ylabel('Time to Retraction (Months)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Select columns with integer or categorical data types
discrete_features = df.select_dtypes(include=['int64', 'object'])

# Exclude date columns from the list
discrete_features = discrete_features.drop(columns=['RetractionDate', 'OriginalPaperDate'], errors='ignore')

# Display the list of discrete features
print("Discrete Features (excluding 'RetractionDate' and 'OriginalPaperDate'):")
print(discrete_features.columns)

# Select columns with numeric data types excluding date columns
continuous_features = df.select_dtypes(include=['float64', 'int64'])

# Exclude date columns from the list if present
continuous_features = continuous_features.drop(columns=['RetractionDate', 'OriginalPaperDate'], errors='ignore')

# Display the list of continuous features
print("Continuous Features (excluding 'RetractionDate' and 'OriginalPaperDate'):")
print(continuous_features.columns)
    
# Drop the record ID column if it exists
if 'RecordID' in df.columns:
    df.drop('RecordID', axis=1, inplace=True)

# Exclude the 'time_to_retraction_months' column from continuous features
continuous_features = continuous_features.drop(columns=['time_to_retraction_months','RecordID'], errors='ignore')

# Plot histograms for continuous features (excluding 'time_to_retraction_months')
for feature in continuous_features.columns:
    plt.figure(figsize=(8, 6))
    plt.hist(df[feature], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'{feature} Distribution')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Apply log transformation to continuous variables and response variable
log_continuous_features = np.log1p(continuous_features)
log_time_to_retraction = np.log1p(df['time_to_retraction_months'])

# Plot scatter plots between each transformed continuous variable and transformed response variable
for feature in log_continuous_features.columns:
    plt.figure(figsize=(8, 6))
    plt.scatter(log_continuous_features[feature], log_time_to_retraction, alpha=0.5)
    plt.title(f'log{feature} vs Log Time to Retraction')
    plt.xlabel('log(feature)')
    plt.ylabel('Log Time to Retraction')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Plot box plots to visualize outliers in log continuous variables
for feature in log_continuous_features.columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=log_continuous_features[feature], color='skyblue')
    plt.title(f'Box Plot of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

from sklearn.model_selection import train_test_split

# Features (excluding 'time_to_retraction_months')
features = df.drop(['time_to_retraction_months'], axis=1)

# Response variable
response_variable = df['time_to_retraction_months']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, response_variable, test_size=0.2, random_state=42)

# Get numerical columns with missing values
numerical_columns_with_missing = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).isnull().any()]

# Print numerical columns with missing values
print("Numerical Columns with Missing Values:")
print(numerical_columns_with_missing)

# Replace missing values in numerical columns with the median of each column
numerical_columns = df.select_dtypes(include=np.number).columns
df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].median())

# Check missing values in numerical columns
missing_values_numerical = df.select_dtypes(include=np.number).isnull().sum()
print("Missing Values in Numerical Columns:")
print(missing_values_numerical)

# Plot the features with year information with respect to retraction time
for feature in features_with_year:
    # Calculate the difference in months between the feature and retraction date
    df[f'{feature}_months_before_retraction'] = (df['RetractionDate'] - df[feature]).dt.days / 30
    
    # Print the newly created column
    print(f"{feature} months before retraction:")
    print(df[f'{feature}_months_before_retraction'].head())
    
    plt.figure(figsize=(8, 6))
    plt.scatter(df[f'{feature}_months_before_retraction'], df['time_to_retraction_months'], alpha=0.5)
    plt.title(f'{feature} vs Time to Retraction')
    plt.xlabel(f'Months Before Retraction based on {feature}')
    plt.ylabel('Time to Retraction (Months)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




