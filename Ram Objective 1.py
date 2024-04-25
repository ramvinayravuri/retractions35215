import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming both snippets refer to the same dataset but with possibly different filenames. Use the correct filename:
# file_path = 'retractions35215.csv'
# Load the dataset
data = pd.read_csv('D:\\PRT564 DATA ANALYTICS AND VISUALISATION\\Group Project\\Objective 1\\retractions35215.csv')

# Display the first few rows to understand the dataset's structure
print("First few rows of the dataset:")
print(data.head())

# General information about data types and missing values
print("\nDataset information:")
print(data.info())

# Summary statistics for numerical data to check for any anomalies
print("\nSummary statistics for numerical columns:")
print(data.describe())

# Check for missing values in each column
print("\nMissing values in each column:")
print(data.isnull().sum())

# Optional: Display unique values for categorical fields to identify inconsistencies
print("\nUnique values in columns:")
for column in data.columns:
    print(f"Unique values in '{column}':")
    print(data[column].dropna().unique())

# Data cleanup for visualization and analysis
# Ensure that 'Notes' and 'RetractionNature' columns are treated as strings
data['Notes'] = data['Notes'].astype(str)
data['RetractionNature'] = data['RetractionNature'].astype(str)

# Check if 'Notes' contains 'Retraction Nature' for each row
data['Has_Retraction_Nature'] = data.apply(lambda x: x['RetractionNature'] in x['Notes'], axis=1)

# Check if 'Notes' contains 'Paywalled' for each row
data['Has_Paywalled'] = data.apply(lambda x: x['Paywalled'] in x['Notes'], axis=1)

# Check if 'Notes' contains 'Reason' for each row
data['Has_Reason'] = data.apply(lambda x: x['Reason'] in x['Notes'], axis=1)

# Creating a new DataFrame to summarize the findings
summary_df = pd.DataFrame({
    'Has Retraction Nature in Notes': data['Has_Retraction_Nature'].value_counts(),
    'Has Paywalled in Notes': data['Has_Paywalled'].value_counts(),
    'Has Reason in Notes': data['Has_Reason'].value_counts()
})

# Display the summary DataFrame
print(summary_df)

# Setting up the aesthetic style for the plots
sns.set(style="whitegrid")

# Plot the top 10 most common reasons for retractions
plt.figure(figsize=(10, 6))
reason_counts = data['Reason'].value_counts().head(10)
sns.barplot(y=reason_counts.index, x=reason_counts.values, palette="viridis")
plt.title('Top 10 Reasons for Retractions')
plt.xlabel('Number of Retractions')
plt.ylabel('Reason')
plt.show()

# Plot the distribution of Paywalled vs Non-Paywalled retractions
plt.figure(figsize=(6, 4))
sns.countplot(x='Paywalled', data=data, palette="viridis")
plt.title('Distribution of Paywalled vs Non-Paywalled Retractions')
plt.xlabel('Paywalled')
plt.ylabel('Count')
plt.show()