#!/usr/bin/env python
# coding: utf-8

# # Stack-Overflow Developer survey dataset Analysis

# we'll analyze the StackOverflow developer survey dataset. The dataset contains responses to an annual survey conducted by StackOverflow. You can find the raw data & official analysis here: https://insights.stackoverflow.com/survey.

# In[2]:


import pandas as pd


# In[3]:


survey_raw_df=pd.read_csv('survey_results_public.csv')


# In[4]:


survey_raw_df


# The dataset contains over 73,000 responses to 78 questions (although many questions are optional). The responses have been anonymized to remove personally identifiable information, and each respondent has been assigned a randomized respondent ID.
# 
# Let's view the list of columns in the data frame.

# In[5]:


survey_raw_df.columns


# It appears that shortcodes for questions have been used as column names.
# 
# We can refer to the schema file to see the full text of each question. The schema file contains only two columns: Column and QuestionText. We can load it as Pandas Series with Column as the index and the QuestionText as the value

# In[14]:


schema_fname='survey_results_schema.csv'
schema_raw=pd.read_csv(schema_fname,index_col='qname').question


# In[15]:


schema_raw


# We can now use schema_raw to retrieve the full question text for any column in survey_raw_df

# In[16]:


schema_raw['Employment']


# We've now loaded the dataset. We're ready to move on to the next step of preprocessing & cleaning the data for our analysis

# # Data Preparation & Cleaning

# While the survey responses contain a wealth of information, we'll limit our analysis to the following areas:
# 
# Demographics of the survey respondents and the global programming community,
# Distribution of programming skills, experience, and preferences,
# Employment-related information, preferences, and opinions
# 
# Let's select a subset of columns with the relevant data for our analysis.

# In[59]:


selected_columns = [
    # Demographics
    'Country',
    'Gender',
    'EdLevel',
    'Ethnicity',
    # Programming experience
    'CodingActivities',
    'YearsCode',
    'YearsCodePro',
    'LanguageHaveWorkedWith',
    'LanguageWantToWorkWith',
    'DatabaseHaveWorkedWith',
    'DatabaseWantToWorkWith',
    'PlatformHaveWorkedWith',
    'PlatformWantToWorkWith',
    # Employment
    'Employment',
    'DevType',
    'TimeSearching',
    'TimeAnswering',
    'Onboarding',
    'RemoteWork',
    'CompTotal'
]


# In[60]:


len(selected_columns)


# Let's extract a copy of the data from these columns into a new data frame survey_df. We can continue to modify further without affecting the original data frame.

# In[61]:


survey_df=survey_raw_df[selected_columns].copy()


# Let's view some basic information about the data frame.

# In[62]:


survey_df.shape


# In[54]:


survey_df.info()


# Most columns have the data type object, either because they contain values of different types or contain empty values (NaN). It appears that every column contains some empty values since the Non-Null count for every column is lower than the total number of rows (73268). We'll need to deal with empty values and manually adjust the data type for each column on a case-by-case basis.
# 
# Only one of the columns was detected as numeric columns (CompTotal), even though a few other columns have mostly numeric values. To make our analysis easier, let's convert some other columns into numeric data types while ignoring any non-numeric value. The non-numeric are converted to NaN.

# In[63]:


survey_df['YearsCode'] = pd.to_numeric(survey_df.YearsCode, errors='coerce')
survey_df['YearsCodePro'] = pd.to_numeric(survey_df.YearsCodePro, errors='coerce')


# Let's now view some basic statistics about numeric columns.

# In[64]:


survey_df.describe()


# The gender column also allows for picking multiple options. We'll remove values containing more than one option to simplify our analysis

# In[65]:


survey_df['Gender'].value_counts()


# In[66]:


import numpy as np


# In[67]:


survey_df.where(~(survey_df.Gender.str.contains(';', na=False)), np.nan, inplace=True)


# 
# We've now cleaned up and prepared the dataset for analysis. Let's take a look at a sample of rows from the data frame.

# In[68]:


survey_df.sample(10)


# # Exploratory Analysis and Visualization

# Before we ask questions about the survey responses, it would help to understand the respondents' demographics, i.e., country, gender, education level, employment level, etc. It's essential to explore these variables to understand how representative the survey is of the worldwide programming community. A survey of this scale generally tends to have some selection bias.
# 
# Let's begin by importing matplotlib.pyplot and seaborn.

# In[72]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

sns.set_style('darkgrid')
matplotlib.rcParams['font.size']=14
matplotlib.rcParams['figure.figsize']=(9,5)
matplotlib.rcParams['figure.facecolor']='#00000000'


# # Country

# Let's look at the number of countries from which there are responses in the survey and plot the ten countries with the highest number of responses.

# In[75]:


schema_raw.Country


# In[76]:


survey_df.Country.nunique()


# We can identify the countries with the highest number of respondents using the value_counts method.

# In[78]:


top_countries=survey_df.Country.value_counts().head(15)
top_countries


# In[85]:


plt.figure(figsize=(12,6))
plt.xticks(rotation=90)
plt.title(schema_raw.Country)
sns.barplot(x=top_countries.index,y=top_countries);


# It appears that a disproportionately high number of respondents are from the US and India, probably because the survey is in English, and these countries have the highest English-speaking populations. We can already see that the survey may not be representative of the global programming community - especially from non-English speaking countries. Programmers from non-English speaking countries are almost certainly underrepresented.

# # Years Code

# In[87]:


plt.figure(figsize=(12, 6))
plt.title(schema_raw.YearsCode)
plt.xlabel('YearsCode')
plt.ylabel('Number of respondents')

plt.hist(survey_df.YearsCode, bins=np.arange(10,80,5), color='purple');


# We can chech how many years they coded professionaly.

# In[88]:


plt.figure(figsize=(12, 6))
plt.title(schema_raw.YearsCodePro)
plt.xlabel('YearsCodePro')
plt.ylabel('Number of respondents')

plt.hist(survey_df.YearsCodePro, bins=np.arange(10,80,5), color='purple');


# It appears that a large percentage of respondents have 10 years of coding journey . It's somewhat representative of the programming community in general. Many young people have taken up computer science as their field of study or profession in the last 20 years.

# # Gender

# Let's look at the distribution of responses for the Gender. It's a well-known fact that women and non-binary genders are underrepresented in the programming community, so we might expect to see a skewed distribution here.

# In[89]:


schema_raw.Gender


# In[90]:


gender_counts=survey_df.Gender.value_counts()
gender_counts


# A pie chart would be a great way to visualize the distribution.

# In[93]:


plt.figure(figsize=(12,6))
plt.title(schema_raw.Gender)

plt.pie(gender_counts,labels=gender_counts.index,autopct='%1.1f%%',startangle=180);


# Only about 8% of survey respondents who have answered the question identify as women or non-binary. This number is lower than the overall percentage of women & non-binary genders in the programming community - which is estimated to be around 12%.

# # Education Level

# Formal education in computer science is often considered an essential requirement for becoming a programmer. However, there are many free resources & tutorials available online to learn programming. Let's compare the education levels of respondents to gain some insight into this. We'll use a horizontal bar plot here.

# In[94]:


sns.countplot(y=survey_df.EdLevel)
plt.xticks(rotation=75)
plt.title(schema_raw.EdLevel)
plt.ylabel(None);


# It appears that well over half of the respondents hold a bachelor's or master's degree, so most programmers seem to have some college education. However, it's not clear from this graph alone if they hold a degree in computer science.

# # Employment

# Freelancing or contract work is a common choice among programmers, so it would be interesting to compare the breakdown between full-time, part-time, and freelance work. Let's visualize the data from the Employment column.

# In[95]:


schema_raw.Employment


# The Employment column also allows for picking multiple options. We'll remove values containing more than one option to simplify our analysis

# In[103]:


survey_df.where(~(survey_df.Employment.str.contains(';', na=False)), np.nan, inplace=True)


# In[107]:


survey_df.Employment.value_counts()


# In[108]:


(survey_df.Employment.value_counts(normalize=True, ascending=True)*100).plot(kind='barh', color='g')
plt.title(schema_raw.Employment)
plt.xlabel('Percentage');


# It appears that close to 10% of respondents are employed part time or as freelancers.

# The DevType field contains information about the roles held by respondents. Since the question allows multiple answers, the column contains lists of values separated by a semi-colon ;, making it a bit harder to analyze directly.

# In[109]:


survey_df.where(~(survey_df.DevType.str.contains(';', na=False)), np.nan, inplace=True)


# In[114]:


dev_count=survey_df.DevType.value_counts()
dev_count


# As one might expect, the most common roles include "Developer" in the name

# # Most popular programming language

# In[116]:


survey_df.LanguageHaveWorkedWith


# Let's define a helper function that turns a column containing lists of values (like survey_df.DevType) into a data frame with one column for each possible option.

# In[118]:


def split_multicolumn(col_series):
    result_df = col_series.to_frame()
    options = []
    # Iterate over the column
    for idx, value  in col_series[col_series.notnull()].iteritems():
        # Break each value into list of options
        for option in value.split(';'):
            # Add the option as a column to result
            if not option in result_df.columns:
                options.append(option)
                result_df[option] = False
            # Mark the value in the option column as True
            result_df.at[idx, option] = True
    return result_df[options]


# In[120]:


languages_worked_df=split_multicolumn(survey_df.LanguageHaveWorkedWith)
languages_worked_df


# In[121]:


languages_worked_percentage=languages_worked_df.mean().sort_values(ascending=False)*100
languages_worked_percentage


# In[122]:


plt.figure(figsize=(12,12))
sns.barplot(x=languages_worked_percentage,y=languages_worked_percentage.index)
plt.title("Languages used in the past year")
plt.xlabel('Count');


# Perhaps unsurprisingly, Javascript & HTML/CSS comes out at the top as web development is one of today's most sought skills. It also happens to be one of the easiest to get started. SQL is necessary for working with relational databases, so it's no surprise that most programmers work with SQL regularly. Python seems to be the popular choice for other forms of development, beating out Java, which was the industry standard for server & application development for over two decades.

# # Most people interested to learn 

# we can use the `LanguageWantToWorkWith` column

# In[124]:


languages_want_work_df=split_multicolumn(survey_df.LanguageWantToWorkWith)
languages_want_work_df


# In[127]:


language_want_work_percent=languages_want_work_df.mean().sort_values(ascending=False)*100


# In[128]:


language_want_work_percent


# In[129]:


plt.figure(figsize=(12,12))
sns.barplot(x=language_want_work_percent,y=language_want_work_percent.index)
plt.title("Languages people are intersted in learning over the next year")
plt.xlabel('Count')


#  It's not surprising that JavaScript is the language most people are interested in learning - since it is an easy-to-learn general-purpose programming language well suited for a variety of domains. JavaScript is a scripting language that enables you to create dynamically updating content, control multimedia, animate images, and pretty much everything else.

# Which are the most loved languages, i.e., a high percentage of people who have used the language want to continue learning & using it over the next year?

# In[136]:


languages_loved_df=languages_worked_df & languages_want_work_df


# In[134]:


languages_loved_percent=(languages_loved_df.sum()*100/languages_worked_df.sum()).sort_values(ascending=False)
languages_loved_percent


# In[133]:


plt.figure(figsize=(12,12))
sns.barplot(x=languages_loved_percent,y=languages_loved_percent.index)
plt.title("Most loved language")
plt.xlabel("Count")


# Rust has been StackOverflow's most-loved language for four years in a row. The second most-loved language is Clojure, a popular alternative to JavaScript for web development.
# 
# Python features at number 6 , despite already being one of the most widely-used languages in the world. Python has a solid foundation, is easy to learn & use, has a large ecosystem of domain-specific libraries, and a massive worldwide community.
# 
# 

# # Most Popular Database

# In[137]:


database_worked_df=split_multicolumn(survey_df.DatabaseHaveWorkedWith)
database_worked_df


# In[138]:


database_worked_percent=database_worked_df.mean().sort_values(ascending=False)*100
database_worked_percent


# In[139]:


plt.figure(figsize=(12,12))
sns.barplot(x=database_worked_percent,y=database_worked_percent.index)
plt.title("Databases used in the past year")
plt.xlabel('Count');


# # Database Want to Work With

# In[140]:


database_want_work_df=split_multicolumn(survey_df.DatabaseWantToWorkWith)
database_want_work_df


# In[142]:


database_want_percent=database_want_work_df.mean().sort_values(ascending=False)*100
database_want_percent


# In[144]:


plt.figure(figsize=(12,12))
sns.barplot(x=database_want_percent,y=database_want_percent.index)
plt.title("Databases want to work in next year ")
plt.xlabel('Count');


# # Most loved Database

# In[145]:


database_loved_df=database_worked_df & database_want_work_df


# In[146]:


database_loved_percent=(database_loved_df.sum()*100/database_worked_df.sum()).sort_values(ascending=False)
database_loved_percent


# In[147]:


plt.figure(figsize=(12,12))
sns.barplot(x=database_loved_percent,y=database_loved_percent.index)
plt.title("Most loved database")
plt.xlabel("Count")


# PostgreSQL has been StackOverflow's most-loved database for years in a row. The second most-loved database is Redis, which stands for Remote Dictionary Server, is a fast, open source, in-memory, key-value data store.

# In[ ]:




