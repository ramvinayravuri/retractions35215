import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Displaying all columns of the dataframe
pd.pandas.set_option("display.max_columns", None)
df = pd.read_csv("retractions35215.csv")

# Drop rows with missing values in the "RetractionDOI" column
df = df.dropna(subset=['RetractionDOI'])
df = df.dropna(subset=['RetractionPubMedID'])
df = df.dropna(subset=['OriginalPaperPubMedID'])

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

# Assuming df is your DataFrame
df['RetractionDate'] = pd.to_datetime(df['RetractionDate'], format='%d/%m/%Y')
df['OriginalPaperDate'] = pd.to_datetime(df['OriginalPaperDate'], format='%d/%m/%Y')

# Extract decade from RetractionDate and OriginalPaperDate
df['RetractionDecade'] = df['RetractionDate'].dt.year // 10 * 10
df['OriginalPaperDecade'] = df['OriginalPaperDate'].dt.year // 10 * 10

# Number of retractions per decade
retractions_per_decade = df['RetractionDecade'].value_counts().sort_index()

# Number of original papers per decade
original_papers_per_decade = df['OriginalPaperDecade'].value_counts().sort_index()

# Plotting the number of retractions and original papers per decade
plt.figure(figsize=(12, 6))
plt.plot(retractions_per_decade.index, retractions_per_decade.values, label='Retractions', marker='o')
plt.plot(original_papers_per_decade.index, original_papers_per_decade.values, label='Original Papers', marker='o')
plt.xlabel('Decade')
plt.ylabel('Count')
plt.title('Number of Retractions and Original Papers per Decade')
plt.legend()
plt.grid(True)
plt.xticks(retractions_per_decade.index, [f'{decade}s' for decade in retractions_per_decade.index])
plt.show()

# Subject: Analyze the distribution of retractions across different fields of study
subject_distribution = df['Subject'].value_counts().nlargest(10)

# Plotting the distribution of retractions across different fields of study (Pie Chart)
plt.figure(figsize=(12, 8))
subject_distribution.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Distribution of Retractions Across Different Fields of Study')
plt.tight_layout()
plt.show()

# Journal: Identify journals with a higher frequency of retracted papers
journal_retraction_counts = df['Journal'].value_counts()

# Plotting the journals with a higher frequency of retracted papers (Horizontal Bar Chart)
plt.figure(figsize=(12, 10))
journal_retraction_counts.nlargest(20).plot(kind='barh')
plt.xlabel('Count')
plt.ylabel('Journal')
plt.title('Journals with a Higher Frequency of Retracted Papers')
plt.gca().invert_yaxis()  # Invert y-axis to show the journal with the highest count at the top
plt.tight_layout()
plt.show()

# Number of Authors per Retracted Paper
df['AuthorCount'] = df['Author'].apply(lambda x: min(21, len(x.split(';'))))
author_counts = df['AuthorCount'].value_counts().sort_index()

# Plotting the distribution of number of authors per retracted paper
plt.figure(figsize=(12, 6))
author_counts.plot(kind='bar')
plt.xlabel('Number of Authors')
plt.ylabel('Count')
plt.title('Distribution of Number of Authors per Retracted Paper')
plt.xticks(range(1, 22))
plt.xlim(0.5, 21.5)  # Limiting x-axis to 21 authors
plt.show()

# Prevalence of Retractions by Country
country_counts = df['Country'].value_counts()

# Plotting the distribution of retractions across different countries (Top 10 countries)
plt.figure(figsize=(12, 8))
country_counts.nlargest(10).plot(kind='bar')
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Distribution of Retractions Across Top 10 Countries')
plt.xticks(rotation=45)
plt.show()

# Reasons for Retraction (RetractionNature)
retraction_reasons = df['RetractionNature'].value_counts()

# Correlation between Paywalled Journals and Retractions
paywalled_counts = df['Paywalled'].value_counts()

# Plotting the distribution of retractions in paywalled vs. non-paywalled journals (Horizontal Bar Chart)
plt.figure(figsize=(8, 6))
paywalled_counts.plot(kind='barh', color=['skyblue', 'lightcoral'])
plt.xlabel('Count')
plt.ylabel('Paywalled')
plt.title('Distribution of Retractions in Paywalled vs. Non-Paywalled Journals')
plt.gca().invert_yaxis()  # Invert y-axis to show the paywalled at the top
plt.show()

# Top 10 Publishers
top_publishers = df['Publisher'].value_counts().nlargest(10)

# Top 10 Authors
top_authors = df['Author'].str.split(';').explode().str.strip().value_counts().nlargest(10)

# Plotting the top 10 publishers
plt.figure(figsize=(12, 8))
plt.subplot(1, 2, 1)
top_publishers.plot(kind='barh', color='skyblue')
plt.xlabel('Count')
plt.ylabel('Publisher')
plt.title('Top 10 Publishers with the Most Retractions')
plt.gca().invert_yaxis()

# Plotting the top 10 authors
plt.subplot(1, 2, 2)
top_authors.plot(kind='barh', color='lightcoral')
plt.xlabel('Count')
plt.ylabel('Author')
plt.title('Top 10 Authors with the Most Retractions')
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()













