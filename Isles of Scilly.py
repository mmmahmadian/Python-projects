#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Read the CSV file
df = pd.read_csv('Isles of Scilly 2024.csv')

# Define Likert scale mapping
likert_mapping = {
    'Strongly agree': 100,
    'Agree': 75,
    'Neither agree nor disagree': 50,
    'Disagree': 25,
    'Strongly disagree': 0
}

# Define question labels
question_labels = {'Q1': 'I am confident that I can perform effectively on many different tasks',
    'Q2': 'There is a clear link between my performance and my rewards',
    'Q3': 'Even when things are tough, I can perform well in my job',
    'Q4': 'I understand the support available to enable me to get my job done',
    'Q5': 'I feel able to strongly influence my performance goals',
    'Q6': 'I often “go the extra mile” to get my job done',
    'Q7': 'On the whole, I strive with all my energy to perform my job',
    'Q8': 'At work, I concentrate for long periods on my job',
    'Q9': 'I feel a sense of pride about my job',
    'Q10': 'I constantly experience excessive pressure in my job',
    'Q11': 'I feel secure in my job',
    'Q12': 'I feel my pay and benefits are reasonable for what I do',
    'Q13': 'I feel my pay and benefits are reasonable in comparison with other employee groups in my Council',
    'Q14': 'I can freely share work issues with my colleagues/team members',
    'Q15': 'I am provided with the tools needed to do my job',
    'Q16': 'My commitment to Local Government motivates me to do as well as I can in my job',
    'Q17': 'I actively pursue personal development opportunities offered to me',
    'Q18': 'I plan to remain working for the Council of the Isles of Scilly for at least the next 12 months',
    'Q19': 'I regularly have conversations with my line manager about factors affecting my performance',
    'Q20': 'There is a clear link between my appraisal objectives and my team\'s objectives',
    'Q21': 'There is a strong desire in my team to find smarter ways of doing things',
    'Q22': 'When things are tough, we work well together as a team',
    'Q23': 'I am proud of the work my team delivers to our customers',
    'Q24': 'My line manager recognises that speaking openly about problems in the workplace provides an opportunity to improve things',
    'Q25': 'I am well supported by my line manager most of the time',
    'Q26': 'I am satisfied with the way my appraisal has been conducted by my line manager',
    'Q27': 'My line manager encourages conversations within my team about creating solutions to work-related problems',
    'Q28': 'I have useful conversations with my line manager to find practical solutions to problems I experience at work',
    'Q29': 'My line manager encourages conversations that enable the team to be more effective in achieving its performance goals',
    'Q30': 'I do not hesitate to challenge the opinions of others if I believe it will enhance the work of my team',
    'Q31': 'I am willing to put myself out for my team members when required',
    'Q32': 'I trust my line manager to act in my best interests',
    'Q33': 'My manager encourages me and my colleagues to be flexible about when and where we work and to use space and technology creatively',
    'Q34': 'I feel positive and able to cope with my work most of the time',
    'Q35': 'I feel I am treated fairly and respectfully by my colleagues',
    'Q36': 'I feel physically safe when carrying out my job',
    'Q37': 'My job makes a real difference to other people (colleagues and service users)',
    'Q38': 'At work, I am encouraged to make time for my own wellbeing activities',
    'Q39': 'I have access to facilities (e.g., physical space) to help me rest and recover during my working day',
    'Q40': 'If I have concerns, I feel safe in raising them',
    'Q41': 'My Council gives me opportunities to help design organisational change that affects me',
    'Q42': 'My Council involves me in implementing change that affects me',
    'Q43': 'My Council invests in building my capabilities through learning and development',
    'Q44': 'My Council values my accomplishments at work',
    'Q45': 'My Council demonstrates a genuine concern for my well-being',
    'Q46': 'There is a “no blame” culture – mistakes are talked about freely so we can learn from them',
    'Q47': 'I do not hesitate to challenge the opinion of others if I believe that it will enhance the workings of my Council',
    'Q48': 'I would recommend working for my Council to a friend',
    'Q49': 'I feel that my Council\'s values appeal to my personal values',
    'Q50': 'My Council\'s leadership team have a clear vision for the future of the organisation',
    'Q51': 'My Council\'s leadership team inspire me to use my own initiative',
    'Q52': 'I have a clear view about my Council\'s obligations to me',
    'Q53': 'I trust my Council to deliver on its obligations to me',
    'Q54': 'My Council recognises that speaking openly about workplace problems provides an opportunity to improve things',
    'Q55': 'My Council provides me with good prospects for developing my career',
    'Q56': 'Overall, I am satisfied with the employment deal provided by my Council (what I receive and what I am expected to give in return)',
    'Q57': '1. Compliance with internal procedures often makes it difficult to produce creative solutions',
    'Q58': '2. My personal development preferences are often overridden by the needs of my Council',
    'Q59': '3. The quality of service delivery is often compromised by time pressures',
    'Q60': '4. I am often required to do more with less resources',
    'Q61': '5. The immediate demands of my job often conflict with achieving longer term goals',
    'Q62': 'Conversational - e.g., Respectful, Good Listener',
    'Q63': 'Capable - e.g., Competent, Confident',
    'Q64': 'Innovative - e.g., imaginative, Inspiring',
    'Q65': 'Trustworthy - e.g., Honest, Reliable',
    'Q66': 'Supportive - e.g., Compassionate, Appreciative',
    'Q67': 'Directive - e.g., Authoritarian, Controlling',
    'Q68': '1. The leadership of my workplace encourages and is supportive of Equality, Diversity and Inclusion',
    'Q69': '2. Management shows that Equality, Diversity and Inclusion is important through its actions',
    'Q70': '3. My Council is committed to improving the diversity of its workforce',
    'Q71': '4. My Council fosters a culture that allows staff to be themselves at work without fear',
    'Q72': '5. My Council takes active measures to seek a diverse candidate pool when hiring',
    'Q73': '6. Employees of different backgrounds are encouraged to apply for higher positions in my Council',
    'Q74': '7. My Council\'s policies or procedures encourage Equality, Diversity and Inclusion',
    'Q75': '8. My Council provides an environment for the free and open expression of ideas, opinions, and beliefs.'

}
mean_values = {}
for col in df.columns:
    if df[col].dtype == 'object':  # Check if column contains string values
        df[col] = df[col].map(likert_mapping)
    
    # Skip NaN values when calculating mean
    mean_value = df[col].mean(skipna=True)
    if not np.isnan(mean_value):
        mean_values[question_labels.get(col, col)] = round(mean_value)

# Create DataFrame for mean values
mean_df = pd.DataFrame(list(mean_values.items()), columns=['Questions', 'Mean Value'])

# Remove "respond id" row
mean_df = mean_df[mean_df['Questions'] != 'Respondent ID']

# Define color function
def get_color(value, question):
    if question in ['1. Compliance with internal procedures often makes it difficult to produce creative solutions',
                    '2. My personal development preferences are often overridden by the needs of my Council',
                    '3. The quality of service delivery is often compromised by time pressures',
                    '4. I am often required to do more with less resources',
                    '5. The immediate demands of my job often conflict with achieving longer term goals']:
        if value >= 75:
            return 'red'
        elif value <= 50:
            return 'green'
        else:
            return 'yellow'
    else:
        if value >= 75:
            return 'green'
        elif value >= 50:
            return 'yellow'
        else:
            return 'red'
        

# Create table-like plot
fig, ax = plt.subplots(figsize=(20, 15))
ax.axis('off')

# Create table
table_data = mean_df.values
table = ax.table(cellText=table_data, colLabels=mean_df.columns, loc='center')

# Set header properties
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(mean_df.columns))))

# Set cell colors based on mean values
for i in range(len(mean_df)):
    cell = table[i+1, 1]  # Second column (index 1) for mean values
    value = int(table_data[i, 1])
    question = table_data[i, 0]
    color = get_color(value, question)
    cell.set_facecolor(color)
    
    # Align questions to the left
    cell_question = table[i+1, 0]  # First column (index 0) for questions
    cell_question.set_text_props(ha='left')

plt.tight_layout()
mean_df.to_excel('mean_questions_IOS.xlsx', index=False)
plt.show()
print(df.columns)


# In[10]:


import pandas as pd

# Define an empty list to hold dictionaries of data
mean_data = []

# Define questions for each variable
variables = {
    'PC': ['Q2','Q5','Q11','Q12','Q13','Q15','Q20','Q32','Q50','Q52','Q53','Q55'],
    'POS': ['Q41','Q42','Q43','Q44','Q45','Q46','Q51','Q54'],
    'Employer_Contribution': ['PC', 'POS'],
    'OE': ['Q16','Q17','Q30','Q31','Q47','Q48','Q49'],
    'CAP': ['Q1','Q3','Q21','Q22'],
    'JE': ['Q6','Q7','Q8','Q9','Q14','Q23'],
    'Employee_Contribution': ['OE', 'CAP', 'JE'],
    'Satisfaction': ['Q56'],
    'CP_S': ['Q24','Q27','Q28','Q4','Q33'],
    'CP_P': ['Q19','Q25','Q26','Q29'],
    'CP': ['CP_S', 'CP_P'],
    'JP': ['Q10'],
    'WT': ['Q57','Q58','Q59','Q60','Q61'],
    'Health & Wellbeing': ['Q34','Q35','Q36','Q37','Q38','Q39','Q40'],
    'Desire to stay':['Q18']
}
for variable, questions in variables.items():
    # Check if the variable is grouped
    if variable in ['Employer_Contribution','Employee_Contribution','CP']:
        # Extract the column names from the variables
        question_columns = [variables[question] for question in questions]
        # Flatten the list of column names
        question_columns = [item for sublist in question_columns for item in sublist]
        # Calculate the mean for the grouped columns
        mean_value = df[question_columns].mean(axis=1).mean()
    else:
        # Calculate the mean for the single group of columns
        mean_value = df[questions].mean(axis=1).mean()
    
    # Append the mean value to the DataFrame
    mean_data.append({'Question': variable, 'Mean Value': mean_value})

# Convert the list of dictionaries into a DataFrame
mean_df = pd.DataFrame(mean_data)

# Print the DataFrame
print(mean_df)
# Export the DataFrame to Excel
mean_df.to_excel('IOS_TEDD.xlsx', index=False)

print("Exported DataFrame to 'IOS_TEDD.xlsx'")


# In[11]:


# Read the CSV file
df = pd.read_csv('Isles of Scilly 2024.csv')

services_freq = df['Services'].value_counts().sort_index()

gender_freq = df['Gender'].value_counts().sort_index()

age_freq = df['Age'].value_counts().sort_index()

length_of_services_freq = df['Length of Services'].value_counts().sort_index()

management_level_freq = df['Management level'].value_counts().sort_index()

religion_freq = df['religion'].value_counts().sort_index()

ethnicity_freq = df['Ethnicity'].value_counts().sort_index()

sexual_orientation_freq = df['Sexual orientation'].value_counts().sort_index()

disability_freq = df['disability'].value_counts().sort_index()

# Create a Pandas Excel writer using XlsxWriter as the engine
with pd.ExcelWriter('demographic_frequency_IOS.xlsx', engine='xlsxwriter') as writer:
    # Write each DataFrame to a separate worksheet
    services_freq.to_excel(writer, sheet_name='Services')
    age_freq.to_excel(writer, sheet_name='Age frequency')
    gender_freq.to_excel(writer, sheet_name='Gender frequency')
    length_of_services_freq.to_excel(writer, sheet_name='Length of Services frequency')
    management_level_freq.to_excel(writer, sheet_name='Management level frequency')
    religion_freq.to_excel(writer, sheet_name='Religion frequency')
    ethnicity_freq.to_excel(writer, sheet_name='Ethnicity frequency')
    sexual_orientation_freq.to_excel(writer, sheet_name='Sexual Orientation frequency')
    disability_freq.to_excel(writer, sheet_name='Disability frequency')

print("Data exported to 'demographic_frequency_IOS.xlsx'")


# In[12]:


import pandas as pd

# Apply Likert scale mapping to relevant columns
for col in df.columns:
    if col.startswith('Q'):
        df[col] = df[col].map(likert_mapping)

# Calculate the mean value for each question by gender
mean_values_by_gender = {}
for col in df.columns:
    if col.startswith('Q'):
        mean_values = df.groupby('Gender')[col].mean()
        question_label = question_labels.get(col, col)
        mean_values_by_gender[question_label] = mean_values

# Print the mean values for each question by gender
for question, mean_values in mean_values_by_gender.items():
    print(f"Mean values for '{question}' by gender:")
    print(mean_values)
    print()


# In[13]:


df = pd.read_csv('Isles of Scilly 2024.csv')
# Apply Likert scale mapping to relevant columns

for col in df.columns:
    if col.startswith('Q'):
        df[col] = df[col].map(likert_mapping)

# Define demographic categories
demographics = ['Services', 'Gender', 'Ethnicity', 'disability', 'Management level']

# Initialize dictionary to store mean values by demographic and question
mean_values_by_demographic = {demographic: {} for demographic in demographics}

# Calculate mean values for each demographic and question
for demographic in demographics:
    for col in df.columns:
        if col.startswith('Q'):
            # Calculate mean values for each question within the demographic category
            mean_values = df.groupby(demographic)[col].mean()
            question_label = question_labels.get(col, col)
            mean_values_by_demographic[demographic][question_label] = mean_values

# Create Excel file with separate tabs for each demographic
with pd.ExcelWriter('mean_by_demographics_IOS.xlsx', engine='xlsxwriter') as writer:
    for demographic, mean_values_dict in mean_values_by_demographic.items():
        # Convert mean values dictionary to DataFrame
        mean_values_df = pd.DataFrame(mean_values_dict)
        # Export DataFrame to Excel as a separate tab
        mean_values_df.to_excel(writer, sheet_name=demographic, na_rep='NaN')

print("Mean values by demographics exported to 'mean_by_demographics_IOS.xlsx'")


# In[14]:


import pandas as pd

# Read the CSV file
df = pd.read_csv('Isles of Scilly 2024.csv')

# Apply Likert scale mapping to relevant columns
for col in df.columns:
    if col.startswith('Q'):
        df[col] = df[col].map(likert_mapping)

# Define demographic categories
demographics = ['Services', 'Gender', 'Ethnicity', 'disability', 'Management level']

# Initialize dictionary to store mean values by demographic and question
mean_values_by_demographic = {demographic: {} for demographic in demographics}

# Calculate mean values for each demographic and variable
for demographic in demographics:
    for variable, questions in variables.items():
        # Check if the variable is grouped
        if variable in ['Employer_Contribution', 'Employee_Contribution', 'CP']:
            # Extract the column names from the variables
            question_columns = [variables[question] for question in questions]
            # Flatten the list of column names
            question_columns = [item for sublist in question_columns for item in sublist]
            # Calculate the mean for the grouped columns within the demographic category
            mean_values = df.groupby(demographic)[question_columns].mean().mean(axis=1)
        else:
            # Calculate the mean for the single group of columns within the demographic category
            mean_values = df.groupby(demographic)[questions].mean().mean(axis=1)
        
        # Store the mean value for the demographic and variable
        mean_values_by_demographic[demographic][variable] = mean_values

# Create Excel file with separate tabs for each demographic
with pd.ExcelWriter('Mean_TEDD_IOS.xlsx', engine='xlsxwriter') as writer:
    for demographic, mean_values_dict in mean_values_by_demographic.items():
        # Convert mean values dictionary to DataFrame
        mean_values_df = pd.DataFrame(mean_values_dict)
        # Export DataFrame to Excel as a separate tab
        mean_values_df.to_excel(writer, sheet_name=demographic, na_rep='NaN')

print("Mean values by demographics exported to 'Mean_TEDD_IOS.xlsx'")


# In[15]:


import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('Isles of Scilly 2024.csv')
# Define Likert scale mapping
likert_mapping = {
    'Strongly agree': 100,
    'Agree': 75,
    'Neither agree nor disagree': 50,
    'Disagree': 25,
    'Strongly disagree': 0
}

# Define question labels
question_labels = {'Q1': 'I am confident that I can perform effectively on many different tasks',
    'Q2': 'There is a clear link between my performance and my rewards',
    'Q3': 'Even when things are tough, I can perform well in my job',
    'Q4': 'I understand the support available to enable me to get my job done',
    'Q5': 'I feel able to strongly influence my performance goals',
    'Q6': 'I often “go the extra mile” to get my job done',
    'Q7': 'On the whole, I strive with all my energy to perform my job',
    'Q8': 'At work, I concentrate for long periods on my job',
    'Q9': 'I feel a sense of pride about my job',
    'Q10': 'I constantly experience excessive pressure in my job',
    'Q11': 'I feel secure in my job',
    'Q12': 'I feel my pay and benefits are reasonable for what I do',
    'Q13': 'I feel my pay and benefits are reasonable in comparison with other employee groups in my Council',
    'Q14': 'I can freely share work issues with my colleagues/team members',
    'Q15': 'I am provided with the tools needed to do my job',
    'Q16': 'My commitment to Local Government motivates me to do as well as I can in my job',
    'Q17': 'I actively pursue personal development opportunities offered to me',
    'Q18': 'I plan to remain working for the Council of the Isles of Scilly for at least the next 12 months',
    'Q19': 'I regularly have conversations with my line manager about factors affecting my performance',
    'Q20': 'There is a clear link between my appraisal objectives and my team\'s objectives',
    'Q21': 'There is a strong desire in my team to find smarter ways of doing things',
    'Q22': 'When things are tough, we work well together as a team',
    'Q23': 'I am proud of the work my team delivers to our customers',
    'Q24': 'My line manager recognises that speaking openly about problems in the workplace provides an opportunity to improve things',
    'Q25': 'I am well supported by my line manager most of the time',
    'Q26': 'I am satisfied with the way my appraisal has been conducted by my line manager',
    'Q27': 'My line manager encourages conversations within my team about creating solutions to work-related problems',
    'Q28': 'I have useful conversations with my line manager to find practical solutions to problems I experience at work',
    'Q29': 'My line manager encourages conversations that enable the team to be more effective in achieving its performance goals',
    'Q30': 'I do not hesitate to challenge the opinions of others if I believe it will enhance the work of my team',
    'Q31': 'I am willing to put myself out for my team members when required',
    'Q32': 'I trust my line manager to act in my best interests',
    'Q33': 'My manager encourages me and my colleagues to be flexible about when and where we work and to use space and technology creatively',
    'Q34': 'I feel positive and able to cope with my work most of the time',
    'Q35': 'I feel I am treated fairly and respectfully by my colleagues',
    'Q36': 'I feel physically safe when carrying out my job',
    'Q37': 'My job makes a real difference to other people (colleagues and service users)',
    'Q38': 'At work, I am encouraged to make time for my own wellbeing activities',
    'Q39': 'I have access to facilities (e.g., physical space) to help me rest and recover during my working day',
    'Q40': 'If I have concerns, I feel safe in raising them',
    'Q41': 'My Council gives me opportunities to help design organisational change that affects me',
    'Q42': 'My Council involves me in implementing change that affects me',
    'Q43': 'My Council invests in building my capabilities through learning and development',
    'Q44': 'My Council values my accomplishments at work',
    'Q45': 'My Council demonstrates a genuine concern for my well-being',
    'Q46': 'There is a “no blame” culture – mistakes are talked about freely so we can learn from them',
    'Q47': 'I do not hesitate to challenge the opinion of others if I believe that it will enhance the workings of my Council',
    'Q48': 'I would recommend working for my Council to a friend',
    'Q49': 'I feel that my Council\'s values appeal to my personal values',
    'Q50': 'My Council\'s leadership team have a clear vision for the future of the organisation',
    'Q51': 'My Council\'s leadership team inspire me to use my own initiative',
    'Q52': 'I have a clear view about my Council\'s obligations to me',
    'Q53': 'I trust my Council to deliver on its obligations to me',
    'Q54': 'My Council recognises that speaking openly about workplace problems provides an opportunity to improve things',
    'Q55': 'My Council provides me with good prospects for developing my career',
    'Q56': 'Overall, I am satisfied with the employment deal provided by my Council (what I receive and what I am expected to give in return)',
    'Q57': '1. Compliance with internal procedures often makes it difficult to produce creative solutions',
    'Q58': '2. My personal development preferences are often overridden by the needs of my Council',
    'Q59': '3. The quality of service delivery is often compromised by time pressures',
    'Q60': '4. I am often required to do more with less resources',
    'Q61': '5. The immediate demands of my job often conflict with achieving longer term goals',
    }
mean_values = {}
for col in df.columns:
    if df[col].dtype == 'object':  # Check if column contains string values
        df[col] = df[col].map(likert_mapping)
    
    # Skip NaN values when calculating mean
    mean_value = df[col].mean(skipna=True)
    if not np.isnan(mean_value):
        mean_values[question_labels.get(col, col)] = round(mean_value)

# Create DataFrame for mean values
mean_df = pd.DataFrame(list(mean_values.items()), columns=['Questions', 'Mean Value'])

# Remove "respond id" row
mean_df = mean_df[mean_df['Questions'] != 'Respondent ID']

# Define questions for each variable
variables = {
    'PC': ['Q2','Q5','Q11','Q12','Q13','Q15','Q20','Q32','Q50','Q52','Q53','Q55'],
    'POS': ['Q41','Q42','Q43','Q44','Q45','Q46','Q51','Q54'],
    'Employer_Contribution': ['PC', 'POS'],
    'OE': ['Q16','Q17','Q30','Q31','Q47','Q48','Q49'],
    'CAP': ['Q1','Q3','Q21','Q22'],
    'JE': ['Q6','Q7','Q8','Q9','Q14','Q23'],
    'Employee_Contribution': ['OE', 'CAP', 'JE'],
    'Satisfaction': ['Q56'],
    'CP_S': ['Q24','Q27','Q28','Q4','Q33'],
    'CP_P': ['Q19','Q25','Q26','Q29'],
    'CP': ['CP_S', 'CP_P'],
    'JP': ['Q10'],
    'WT': ['Q57','Q58','Q59','Q60','Q61'],
    'Health & Wellbeing': ['Q34','Q35','Q36','Q37','Q38','Q39','Q40'],
    'Desire to stay':['Q18']
}

mean_data = []
for variable, questions in variables.items():
    if variable in ['Employer_Contribution', 'Employee_Contribution', 'CP']:
        # Extract the column names from the variables
        question_columns = [variables[question] for question in questions]
        # Flatten the list of column names
        question_columns = [item for sublist in question_columns for item in sublist]
        # Calculate the mean for the grouped columns
        mean_value = df[question_columns].mean(axis=1).mean()
    else:
        # Calculate the mean for the single group of columns
        mean_value = df[questions].mean(axis=1).mean()
    
    # Append the mean value to the DataFrame
    mean_data.append({'Question': variable, 'Mean Value': mean_value})

# Convert the list of dictionaries into a DataFrame
mean_df = pd.DataFrame(mean_data)

# Fill missing values with mean
df.fillna(df.mean(), inplace=True)

# Convert the list of dictionaries into a DataFrame
mean_df = pd.DataFrame(mean_data)
df['PC'] = df[['Q2','Q5','Q11','Q12','Q13','Q15','Q20','Q32','Q50','Q52','Q53','Q55']].mean(axis=1)
df['POS'] = df[['Q41','Q42','Q43','Q44','Q45','Q46','Q51','Q54']].mean(axis=1)
df['Employer Contribution']=df[['PC','POS']].mean(axis=1)
df['OE'] = df[['Q16','Q17','Q30','Q31','Q47','Q48','Q49']].mean(axis=1)
df['CAP'] = df[['Q1','Q3','Q21','Q22']].mean(axis=1)
df['JE'] = df[['Q6','Q7','Q8','Q9','Q14','Q23']].mean(axis=1)
df['Employee_Contribution'] = df[['OE', 'CAP', 'JE']].mean(axis=1)
df['Satisfaction'] = df[['Q56']].mean(axis=1)
df['CP_S'] = df[['Q24','Q27','Q28','Q4','Q33']].mean(axis=1)
df['CP_P'] = df[['Q19','Q25','Q26','Q29']].mean(axis=1)
df['CP'] = df[['CP_S', 'CP_P']].mean(axis=1)
df['JP'] = df[['Q10']].mean(axis=1)
df['WT'] = df[['Q57','Q58','Q59','Q60','Q61']].mean(axis=1)
df['Health & Wellbeing'] = df[['Q34','Q35','Q36','Q37','Q38','Q39','Q40']].mean(axis=1)
df['Desire to stay'] = df[['Q18']].mean(axis=1)

# Define the list of predictor questions
predictor_questions = ['Q4','Q33','Q19','Q25','Q26','Q29','Q24','Q27','Q28','Q18','Q34','Q35','Q36','Q37','Q38','Q39','Q40','Q10',
                       'Q56','Q2','Q5','Q11','Q12','Q13','Q15','Q20','Q32','Q50','Q52','Q53','Q55','Q41','Q42','Q43','Q44','Q45',
                       'Q46','Q51','Q54','Q57','Q58','Q59','Q60','Q61'
]

target_variable = 'Employee_Contribution'

# Prepare the data
X = df[predictor_questions]
y = df[target_variable]

X.fillna(X.mean(), inplace=True)

# Now fit the linear regression model
model = sm.OLS(y, sm.add_constant(X)).fit()

# Extract p-values and coefficients
p_values = model.pvalues[1:]  # Exclude the intercept
coefficients = model.params[1:]  # Exclude the intercept

# Create a DataFrame to store the results
results_df = pd.DataFrame({
    'Key drivers of Employee Contribution': [question_labels[predictor] for predictor in X.columns],
    'Impact': coefficients.values * 25,  # Multiply coefficient values by 25
    'P-value': ["{:.2f}".format(p_value) for p_value in p_values.values],  # Format p-values to two decimal points
})

# Remove rows with p-values above 0.05 and sort by Impact
results_df = results_df[results_df['P-value'].astype(float) < 0.05]
results_df = results_df.reindex(results_df['Impact'].abs().sort_values(ascending=False).index)

# Print the results
print("Key drivers of Employee Contribution", "Impact", "P-value", sep="\t")
for index, row in results_df.iterrows():
    print(f"{row['Key drivers of Employee Contribution']:<100} {row['Impact']:.2f} {row['P-value']:<10}")
    
results_df.to_excel('KDA_IOS.xlsx', index=False)   


# In[ ]:




