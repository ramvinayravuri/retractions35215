import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
data = pd.read_csv('D:\\PRT564 DATA ANALYTICS AND VISUALISATION\\Group Project\\Objective 1\\retractions35215.csv')

# Display basic information and the first few rows of the dataset
print(data.info())
print(data.head())

# Checking for missing values in the relevant columns
print("\nMissing values in each column:")
print(data[['Title', 'Subject', 'Institution', 'Publisher', 'Journal']].isnull().sum())

# Fill missing values with a placeholder directly
data.fillna({
    'Title': 'No Title',
    'Subject': 'Unknown Subject',
    'Institution': 'Unknown Institution',
    'Publisher': 'Unknown Publisher',
    'Journal': 'Unknown Journal',
    'Reason': 'Unknown Reason',
}, inplace=True)

# Convert RetractionDate to year if it's not formatted as such
if 'RetractionDate' in data.columns:
    data['RetractionYear'] = pd.to_datetime(data['RetractionDate'], errors='coerce').dt.year

# Display basic information after filling missing values
print(data.head())

# Exploratory Data Analysis (EDA)

# Frequency of retractions per journal
journal_counts = data['Journal'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=journal_counts.head(10).values, y=journal_counts.head(10).index)
plt.title('Top 10 Journals by Retraction Count')
plt.xlabel('Number of Retractions')
plt.ylabel('Journal')
plt.show()

# Frequency of retractions per publisher
publisher_counts = data['Publisher'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=publisher_counts.head(10).values, y=publisher_counts.head(10).index)
plt.title('Top 10 Publishers by Retraction Count')
plt.xlabel('Number of Retractions')
plt.ylabel('Publisher')
plt.show()

# Frequency of retractions per subject
subject_counts = data['Subject'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=subject_counts.head(10).values, y=subject_counts.head(10).index)
plt.title('Top 10 Subjects by Retraction Count')
plt.xlabel('Number of Retractions')
plt.ylabel('Subject')
plt.show()

# Frequency of retractions per institution
institution_counts = data['Institution'].value_counts()
plt.figure(figsize=(10, 6))
sns.barplot(x=institution_counts.head(10).values, y=institution_counts.head(10).index)
plt.title('Top 10 Institutions by Retraction Count')
plt.xlabel('Number of Retractions')
plt.ylabel('Institution')
plt.show()

# Plotting the trend of retractions over time, if 'RetractionYear' was successfully created
if 'RetractionYear' in data.columns:
    plt.figure(figsize=(12, 6))
    data['RetractionYear'].value_counts().sort_index().plot(kind='line')
    plt.title('Trend of Retractions Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Retractions')
    plt.grid(True)
    plt.show()
else:
    print("RetractionYear column not available for trend analysis.")
    
#Time Analysis: Checcking how quickly retractions occur after publication can help identify how swiftly errors are caught and corrected

# The histogram plot will help visualize the distribution of time intervals between the publication and retraction of papers
if 'OriginalPaperDate' in data.columns and 'RetractionDate' in data.columns:
    data['OriginalPaperDate'] = pd.to_datetime(data['OriginalPaperDate'], errors='coerce')
    data['RetractionDate'] = pd.to_datetime(data['RetractionDate'], errors='coerce')
    data['TimeToRetraction'] = (data['RetractionDate'] - data['OriginalPaperDate']).dt.days

    plt.figure(figsize=(12, 6))
    sns.histplot(data['TimeToRetraction'].dropna(), bins=30, kde=True)
    plt.title('Distribution of Time to Retraction')
    plt.xlabel('Days from Publication to Retraction')
    plt.ylabel('Frequency')
    plt.show()

# Group and aggregate the reasons for retractions by frequency across all subjects
reason_counts = data['Reason'].value_counts()

# Consider displaying only the top 10 reasons for simplicity
top_reasons = reason_counts.head(10)

# Create a horizontal bar chart for these top reasons
plt.figure(figsize=(10, 8))
sns.barplot(x=top_reasons.values, y=top_reasons.index, palette='viridis')
plt.title('Top 10 Reasons for Retractions Across All Subjects')
plt.xlabel('Frequency')
plt.ylabel('Reasons')
plt.show()

# Assuming 'Country' is derived from 'Institution' or explicitly mentioned
if 'Country' in data.columns:
    country_retractions = data['Country'].value_counts().head(20)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=country_retractions.values, y=country_retractions.index, palette='viridis')
    plt.title('Top 20 Countries by Number of Retractions')
    plt.xlabel('Number of Retractions')
    plt.ylabel('Country')
    plt.show()
    
#  'CitationCount'  
if 'CitationCount' in data.columns:
    plt.figure(figsize=(12, 6))
    sns.histplot(data['CitationCount'].dropna(), bins=30, color='red')
    plt.title('Distribution of Citation Counts for Retracted Papers')
    plt.xlabel('Citation Count')
    plt.ylabel('Frequency')
    plt.show()
    
    # Histogram to show the distribution of citation counts
plt.figure(figsize=(12, 6))
sns.histplot(data['CitationCount'], bins=50, color='blue', kde=True)  # Using KDE to show the distribution curve
plt.title('Distribution of Citation Counts for Retracted Papers')
plt.xlabel('Citation Count')
plt.ylabel('Frequency')
plt.xlim(0, 500)  # Limiting x-axis to focus on the 0-500 range
plt.grid(True)
plt.show()

# Summary statistics for citation counts
print("Summary Statistics for Citation Counts:")
print(data['CitationCount'].describe())

# Analysis of retractions per publisher over time
if 'RetractionYear' in data.columns:
    publisher_year = data.groupby(['Publisher', 'RetractionYear']).size().unstack(fill_value=0)
    top_publishers = publisher_year.sum(axis=1).nlargest(5).index
    publisher_year.loc[top_publishers].T.plot(kind='line', figsize=(12, 6))
    plt.title('Retraction Trends for Top 5 Publishers Over Time')
    plt.xlabel('Year')
    plt.ylabel('Number of Retractions')
    plt.legend(title='Publisher')
    plt.show()