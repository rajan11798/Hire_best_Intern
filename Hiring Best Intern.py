#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


# In[23]:


# Step 1: Load the dataset
data = pd.read_excel("E:\Hiring Best Intern\Applications_for_Machine_Learning_internship_edited.xlsx")  # Replace "intern_dataset.xlsx" with your actual file path


# In[24]:


# Check the column names in the dataset
print(data.columns)


# In[25]:


# Step 2: Filter available interns
available_interns = data[data['Availability'] == 'Yes, I am available for 3 months starting immediately for a full-time internship.']


# In[35]:


# Step 3: Define weights
weight_skills = 0.4
weight_other_skills = 0.1
weight_degree = 0.1
weight_year_of_graduation = 0.05
weight_pg_performance = 0.1
weight_ug_performance = 0.1
weight_12th_performance = 0.1
weight_10th_performance = 0.05

# Step 4: Normalize data
normalized_data = available_interns.copy()
normalized_data[['Python(out3)', 'MachineLearning(outof3)', 'NaturalLanguageProcessing(NLP)(ouof3)', 'DeepLearning(outof3)']] = available_interns[['Python(out3)', 'MachineLearning(outof3)', 'NaturalLanguageProcessing(NLP)(ouof3)', 'DeepLearning(outof3)']].apply(lambda x: x / 3)



# Step 5: Convert 'Otherskills' into numerical representation using TF-IDF.
# This approach converts each skill into a separate feature and assigns a numerical value indicating the importance of that skill in the dataset.
vectorizer = TfidfVectorizer()
other_skills_matrix = vectorizer.fit_transform(normalized_data['Otherskills'].astype(str))
other_skills_df = pd.DataFrame(other_skills_matrix.toarray(), columns=vectorizer.get_feature_names())

# Step 6: Handle CGPA and percentage values in the last four columns
normalized_data['Performance_PG'] = normalized_data['Performance_PG'].replace('NA', np.nan)
normalized_data['Performance_UG'] = normalized_data['Performance_UG'].replace('NA', np.nan)
normalized_data['Performance_12'] = normalized_data['Performance_12'].replace('NA', np.nan)
normalized_data['Performance_10'] = normalized_data['Performance_10'].replace('NA', np.nan)


def convert_cgpa_to_percentage(cgpa):
    if isinstance(cgpa, float) or isinstance(cgpa, int):
        return cgpa
    if cgpa and '/' in cgpa:
        cgpa = cgpa.split('/')[0].strip()  # Extract the numeric part before '/'
    return float(cgpa) * 10 if cgpa and float(cgpa) <= 10 else np.nan

normalized_data['Performance_PG'] = normalized_data['Performance_PG'].apply(convert_cgpa_to_percentage)
normalized_data['Performance_UG'] = normalized_data['Performance_UG'].apply(convert_cgpa_to_percentage)
normalized_data['Performance_12'] = normalized_data['Performance_12'].apply(convert_cgpa_to_percentage)
normalized_data['Performance_10'] = normalized_data['Performance_10'].apply(convert_cgpa_to_percentage)

# Normalize the values to a common scale (e.g., divide by 100)
normalized_data['Performance_PG'] = normalized_data['Performance_PG'].astype(float) / 100
normalized_data['Performance_UG'] = normalized_data['Performance_UG'].astype(float) / 100
normalized_data['Performance_12'] = normalized_data['Performance_12'].astype(float) / 100
normalized_data['Performance_10'] = normalized_data['Performance_10'].astype(float) / 100



# Step 7: Calculate the intern scores with updated weights
normalized_data['Score'] = (normalized_data['Python(out3)'].astype(float) * weight_skills) +                            (normalized_data['MachineLearning(outof3)'].astype(float) * weight_skills) +                            (normalized_data['NaturalLanguageProcessing(NLP)(ouof3)'].astype(float) * weight_skills) +                            (normalized_data['DeepLearning(outof3)'].astype(float) * weight_skills) +                            (other_skills_df * weight_other_skills).sum(axis=1) +                            (normalized_data['CurrentYearOfGraduation'].astype(float) * weight_year_of_graduation) +                            (normalized_data['Performance_PG'].astype(float) * weight_pg_performance) +                            (normalized_data['Performance_UG'].astype(float) * weight_ug_performance) +                            (normalized_data['Performance_12'].astype(float) * weight_12th_performance) +                            (normalized_data['Performance_10'].astype(float) * weight_10th_performance)

# # Step 5: Calculate scores
# normalized_data['Score'] = (normalized_data['Python(out3)'].astype(float) * weight_skills) + \
#                            (normalized_data['MachineLearning(outof3)'].astype(float) * weight_skills) + \
#                            (normalized_data['NaturalLanguageProcessing(NLP)(ouof3)'].astype(float) * weight_skills) + \
#                            (normalized_data['DeepLearning(outof3)'].astype(float) * weight_skills) + \
#                            (normalized_data['Otherskills'].astype(float) * weight_other_skills) + \
#                            (normalized_data['CurrentYearOfGraduation'].astype(float) * weight_year_of_graduation) + \
#                            (normalized_data['Performance_PG'].astype(float) * weight_pg_performance) + \
#                            (normalized_data['Performance_UG'].astype(float) * weight_ug_performance) + \
#                            (normalized_data['Performance_12'].astype(float) * weight_12th_performance) + \
#                            (normalized_data['Performance_10'].astype(float) * weight_10th_performance)

# Step 8: Rank the interns
ranked_interns = normalized_data.sort_values(by='Score', ascending=False)

# Step 10: you can  filter out scores on diffrent criteria
# target_degree = 'B.tech'  # Specify the desired degree or you can make a dictionory of diffrent degrees
# target_year_of_passing = 2022  # Specify the desired year of passing

# # Step 11: Filter interns based on score, degree, and year of passing
# filtered_interns = ranked_interns[
#     (ranked_interns['Degree'] == target_degree) &
#     (ranked_interns['YearOfGraduation'] == target_year_of_passing)]

# Step 10: Select the best intern
best_intern = ranked_interns.iloc[0]

# Step 11: Print the best intern
print("Best Intern:")
print(best_intern)


# In[38]:


# Step 12: Select the best intern from the filtered interns
best_intern_index = ranked_interns['Score'].idxmax() + 2 #since the index is starting from 2
best_intern = data.loc[best_intern_index]

# Print the details of the best intern
print("Best Intern:")
print(best_intern)

