# Import libraries
import numpy as np
import pandas as pd

import scipy.stats as stats
import random

import matplotlib.pyplot as plt

from pandas_profiling import ProfileReport

'''
Fields

    Postcode: str
        A 6 digit identifier for recognising a place. Often used as key. 
    
    Population: int
    
    
'''

# Generate data

N = 42535
# Postcode
min_postcode = 2000
max_postcode = 2550
poisson_loc = 7500
loc_weight = stats.poisson.pmf(k=list(range(min_postcode, max_postcode)), mu=poisson_loc, loc=min_postcode-poisson_loc) + \
0.05 * stats.binom.pmf(k=list(range(min_postcode, max_postcode)), p=0.5, n=len(list(range(min_postcode, max_postcode))), loc=min_postcode+225) + \
0.08 * stats.binom.pmf(k=list(range(min_postcode, max_postcode)), p=0.5, n=len(list(range(min_postcode, max_postcode))), loc=min_postcode-65) + \
0.05 * stats.binom.pmf(k=list(range(min_postcode, max_postcode)), p=0.5, n=len(list(range(min_postcode, max_postcode))), loc=min_postcode-150) + \
0.035 * stats.binom.pmf(k=list(range(min_postcode, max_postcode)), p=0.5, n=len(list(range(min_postcode, max_postcode))), loc=min_postcode-200)

# Not adding an id col since pandas has one already. Use that one.
# X columns
age_ls = stats.poisson.rvs(mu=25, size=N)
gender_ls = random.choices(['Male', 'Female', 'Other'], weights=[4, 5, 1], k=N)
loc_ls = random.choices(list(range(min_postcode, max_postcode)), weights=loc_weight, k=N)
region_ls = random.choices(['Buddhism', 'Hinduism', 'Islam', 'Christian', 'Catholic', 'Other'], weights=[2, 1, 2, 3, 3, 0.5], k=N)
ethnic_ls = random.choices(['White', 'Aboriginal', 'Asian', 'African', 'Middle East', 'Hispanic', 'Other', 'No'], weights=[8, 2, 3, 1, 4, 3, 3, 20], k=N)

df = pd.DataFrame({'Age': age_ls, 'Gender': gender_ls, 'Location': loc_ls, 'Ethnic': ethnic_ls, 'Religion': region_ls})
location_cat = pd.api.types.CategoricalDtype(categories=list(range(min_postcode, max_postcode)), ordered=True)
df['Location'] = df['Location'].astype(location_cat)

plt.title("Distribution of respondents")
plt.xlabel("Postcode")
plt.ylabel("People")
plt.plot(list(range(min_postcode, max_postcode)), loc_weight * N)
plt.show()

'''
Cultural diversity
'''


def citizen_corr(df):
    df.loc[df['Ethnic'] == 'Aboriginal', 'Citizen'] = ['Yes' for x in df.loc[df['Ethnic'] == 'Aboriginal', 'Citizen']]


def english_prof(df):
    '''
    0 - Does not understand English
    1 - Understand limited English
    2 - Able to speak some English
    3 - Able to speak English
    5 - Can only speak English
    '''

    def gen_english_prof(df, w):
        return random.choices(list(range(0, 5)), weights=w, k=df.shape[0])

    df['English Proficiency'] = np.zeros(df.shape[0])
    df.loc[df['Ethnic'] == 'White', 'English Proficiency'] = 4
    df.loc[df['Ethnic'] == 'Asian', 'English Proficiency'] = gen_english_prof(
        df.loc[df['Ethnic'] == 'Asian', 'English Proficiency'], [5, 3, 1, 1, 3])
    df.loc[df['Ethnic'] == 'Aboriginal', 'English Proficiency'] = gen_english_prof(
        df.loc[df['Ethnic'] == 'Aboriginal', 'English Proficiency'], [1, 0, 0, 9, 0])
    df.loc[df['Ethnic'] == 'African', 'English Proficiency'] = gen_english_prof(
        df.loc[df['Ethnic'] == 'African', 'English Proficiency'], [2, 3, 2, 3, 0.2])
    df.loc[df['Ethnic'] == 'Middle East', 'English Proficiency'] = gen_english_prof(
        df.loc[df['Ethnic'] == 'Middle East', 'English Proficiency'], [2, 5, 2, 4, 0.5])
    df.loc[df['Ethnic'] == 'Hispanic', 'English Proficiency'] = gen_english_prof(
        df.loc[df['Ethnic'] == 'Hispanic', 'English Proficiency'], [3, 2, 2, 2, 0.01])
    df.loc[df['Ethnic'] == 'Other', 'English Proficiency'] = gen_english_prof(
        df.loc[df['Ethnic'] == 'Other', 'English Proficiency'], [4, 3, 3, 2, 0])


def marriage_status(df):
    df['Maternity'] = 'Single'

    def gen_marriage_status(df, w):
        return random.choices(['Single', 'Married', 'Separated', 'Divorced', 'Widowed'], weights=w, k=df.shape[0])

    df.loc[df['Age'] >= 16, 'Maternity'] = gen_marriage_status(df.loc[df['Age'] >= 16, 'Maternity'],
                                                               [3, 3, 0.5, 1, 0.5])
    less_than20_cond = ((df['Age'] > 16) | (df['Age'] < 20)) & ~(
                (df['Maternity'] == 'Single') | (df['Maternity'] == 'Married'))
    df.loc[less_than20_cond, 'Maternity'] = gen_marriage_status(df[less_than20_cond], [9.2, 1, 0.02, 0, 0])
    less_than_25_cond = ((df['Age'] > 20) | (df['Age'] <= 25)) & ~(
                (df['Maternity'] == 'Single') | (df['Maternity'] == 'Married'))
    df.loc[less_than_25_cond, 'Maternity'] = gen_marriage_status(df[less_than_25_cond], [9, 1, 0, 0, 0])


def religion_corr(df):
    df.loc[df['Ethnic'] == 'Aboriginal', 'Religion'] = random.choices(['Christian', 'Catholic', 'Other', 'No'],
                                                                      weights=[3, 1, 2, 4], k=df.loc[
            df['Ethnic'] == 'Aboriginal', 'Religion'].shape[0])


citizenship_ls = random.choices(['Yes', 'No'], weights=[8, 2], k=N)
df['Citizen'] = citizenship_ls
citizen_corr(df)

english_prof(df)
marriage_status(df)
religion_corr(df)

'''
Paid Work
'''

df['Working'] = random.choices(['Yes', 'No'], weights=[6, 4], k=N)
df.loc[df['Age'] < 15, 'Working'] = 'No'

# Self employed
df['Self Employed'] = ['No' for x in df['Age']]  # Age col is to filter one col to fill, does not to be specific
df.loc[df['Working'] == 'Yes', 'Self Employed'] = random.choices(['Yes', 'No'], weights=[3, 7], k=df.loc[df['Working'] == 'Yes', 'Self Employed'].shape[0])
df.loc[df['Working'] == 'No', 'Self Employed'] = random.choices(['Yes', 'No'], weights=[0.05, 9.95], k=df.loc[df['Working'] == 'No', 'Self Employed'].shape[0])

df['Owned Entity'] = ['No' for x in df['Age']]
df.loc[df['Self Employed'] == 'Yes', 'Owned Entity'] = random.choices(['Unincorporated', 'Incorporated'], weights=[9.5, 0.5], k=df.loc[df['Self Employed'] == 'Yes', 'Owned Entity'].shape[0])

# Hours of work
df['Hours of Work'] = [0 for x in df['Age']]
df.loc[df['Working'] == 'Yes', 'Hours of Work'] = stats.norm.rvs(loc=35, scale=5, size=df.loc[df['Working'] == 'Yes', 'Hours of Work'].shape[0])

# Unemployment
df['Finding Work'] = ['No' for x in df['Age']] # By default, no one is finding work until correction
df.loc[df['Working'] == 'No', 'Finding Work'] = random.choices(['Yes', 'No'], weights=[9, 1], k=df.loc[df['Working'] == 'No', 'Finding Work'].shape[0])

'''
Unpaid Work and Care
'''

# Care of children
df['Care of Children'] = stats.binom.rvs(n=40, p=0.35, size=df.shape[0])
df.loc[df['Maternity'] != 'Single', 'Care of Children'] = stats.binom.rvs(n=40, p=0.85, size=df.loc[df['Maternity'] != 'Single', 'Care of Children'].shape[0])
df.loc[df['Maternity'] == 'Widowed', 'Care of Children'] = stats.binom.rvs(n=8, p=0.75, size=df.loc[df['Maternity'] == 'Widowed', 'Care of Children'].shape[0])

# Care of illed family, elderly
df['Care of Family'] = stats.binom.rvs(n=6, p=0.25, size=df.shape[0])
df.loc[df['Maternity'] == 'Widowed', 'Care of Family'] = stats.binom.rvs(n=2, p=0.75, size=df.loc[df['Maternity'] == 'Widowed', 'Care of Family'].shape[0])


# Domestic activities
df['Domestic Activities Hours'] = stats.binom.rvs(n=8, p=0.45, size=df.shape[0])


# Volunteer
df['Volunteered Hours'] = stats.binom.rvs(n=40, p=0.2, size=df.shape[0])

'''
Education
'''
df['Education Level'] = np.floor(stats.norm.rvs(loc=df['Age'].mean(), scale=5, size=df.shape[0])-df['Age'].mean()) + np.floor((2 * (df['English Proficiency'] + 1) + np.where(df['Finding Work'] == "No" , 1, 0) + np.where(df['Gender'] == "Male" , 0.75, 0.72))/13 * 10)
df.loc[df['Education Level'] >= 10, 'Education Level'] = 10
df.loc[df['Education Level'] < 1, 'Education Level'] = 1

'''
Family
'''
df['Family'] = [0 for x in df['Age']]  # By default until correction, members other than self

df.loc[df['Maternity'] != "Single", 'Family'] = df.loc[df['Maternity'] != "Single", 'Family'] + 1

df.loc[df['Care of Children'] > 0, 'Family'] = df.loc[df['Care of Children'] > 0, 'Family'] + stats.binom.rvs(n=3, p=0.25, size=df.loc[df['Care of Children'] > 0, 'Family'].shape[0])

'''
Dwelling
'''

'''
Ownership

Options: 'Owned outright', 'With mortgage', 'Rented', 'Rent-free', 'Other'
'''

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def invlogit(df, intercept, age, english, self_employed, family):
    return sigmoid(intercept + age * ((df['Age']-df['Age'].mean())**2 - df['Age'].mean()) + english * df['English Proficiency'] + self_employed * np.select([df['Self Employed'] == 'Yes'], [1]) + family * (df['Family']+1))

df['Home Internet'] = random.choices(['Yes', 'No'], weights=[99, 1], k=N)

prob_outright = invlogit(df, -1.22, 0.05, 0.019, 0.11, 0.21)
prob_mortgage = invlogit(df, -0.5, 0.04, -0.05, -0.09, 0.5)
prob_rent = invlogit(df, -1.75, 0.1, 0.02, -0.65, 0.001)
prob_rent_free = invlogit(df, -1.88, 0.012, 0.14, -0.65, 0.0005)
prob_other = invlogit(df, -1.5, 0.04, -0.05, -0.09, 0.01)

prob_dwell_type = pd.DataFrame({'Outright': prob_outright, 'Mortgage': prob_mortgage, 'Rent': prob_rent, 'Rent Free': prob_rent_free, 'Other': prob_other})

dwell_type_ls = []
for i in range(N):
    dwell_type_ls.append(random.choices(['Owned outright', 'With mortgage', 'Rented', 'Rent-free', 'Other'], weights=prob_dwell_type.iloc[i,:],k=1)[0])
df['Dwell Type'] = dwell_type_ls

'''
Salary
'''

df['Salary'] = [0 for x in df['Age']]

df.loc[df['Finding Work'] != 'Yes', 'Salary'] = 10000 * (2.35 - 0.05 * ((df['Age']-df['Age'].mean())**2 - df['Age'].mean()) + 0.012 * df['English Proficiency'] - 0.42 * df['Education Level'] + 0.25 * df['Domestic Activities Hours'] + 0.62 * np.select([df['Home Internet'] == 'Yes'], [1]) + 0.215 * np.select([df['Ethnic'] == 'Asian'], [1]) + 0.315 * np.select([df['Ethnic'] == 'White'], [1]) - 0.19 * np.select([df['Ethnic'] == 'Hispanic'], [1]) - 0.201 * np.select([df['Ethnic'] == 'African'], [1]) - 0.12 * np.select([df['Ethnic'] == 'Aboriginal'], [1]) + 0.022 * np.select([df['Ethnic'] == 'Middle East'], [1])).round(2)
df.loc[df['Salary'] < 0, 'Salary'] = 10000 * stats.poisson.rvs(mu=4, size=df.loc[df['Salary'] < 0, 'Salary'].shape[0])

profile = ProfileReport(df, title="Pandas Profiling Report", samples=None, missing_diagrams={"bar": True, "heatmap": False, "dendrogram": True}, explorative=True)
profile.to_file("census_ind.html")

'''
Life Satisfaction Survey
'''
# Generate sample

n = 275
df_life = df.sample(n=n)

'''
Values of life
'''
intercept = -0.80
coef_age = 0.192
coef_gender_male = -0.52
coef_gender_female = 0.0012
coef_citizen = 0.0000015
coef_dwell_owned = 0.12
coef_rent_free = -1.0026
coef_rented = 0.051
coef_mortgage = 0.0023

value_family = (intercept + coef_age * df_life["Age"] + coef_gender_male * df_life["Gender_Male"] + coef_gender_female * df_life["Gender_Female"] + coef_citizen * df_life["Citizen_Yes"] + coef_dwell_owned * df_life["Dwell Type_Owned outright"] + coef_rent_free * df_life["Dwell Type_Rent-free"] + coef_rented * df_life["Dwell Type_Rented"] + coef_mortgage * df_life["Dwell Type_With mortgage"] + stats.beta.rvs(a=2,b=2,loc=-0.5,size=df_life.shape[0])).clip(1, 5)
df_life['VALUE_FAMILY'] = np.round(value_family)

'''
Values of friends
'''
intercept = 1.80
coef_age_squared = -1.35
coef_gender_male = 5.5
coef_gender_female = 3.2011
coef_domestic = -3.21
coef_care_family = -0.15
coef_care_child = -1.13
coef_education = 5.33

value_friends = (intercept + coef_age_squared * (df_life["Age"] - df_life["Age"].mean())**2 + coef_gender_male * df_life["Gender_Male"] + coef_gender_female * df_life["Gender_Female"] + coef_domestic * df_life["Domestic Activities Hours"] + coef_care_family * df_life["Care of Family"] + coef_care_child * df_life["Care of Children"] + coef_education * df_life["Education Level"] + stats.beta.rvs(a=2,b=2,loc=-0.5,size=df_life.shape[0])).clip(1, 5)
df_life['VALUE_FRIENDS'] = np.round(value_friends)


'''
Values of work
'''
intercept = 0.80
coef_age = 1.62
coef_gender_male = 1.42
coef_domestic = -4.21
coef_care_family = 1.2031
coef_care_child = 0.985
coef_finding_work = 2.103
coef_dwell_owned = -4.12
coef_rented = 2.99
coef_mortgage = 1.46

value_work = (intercept + coef_age_squared * abs(df_life["Age"] - 24.5) + coef_gender_male * df_life["Gender_Male"] + coef_domestic * df_life["Domestic Activities Hours"] + coef_care_family * df_life["Care of Family"] + coef_care_child * df_life["Care of Children"] + coef_finding_work * df_life["Finding Work_Yes"] + coef_dwell_owned * df_life["Dwell Type_Owned outright"] + coef_rented * df_life["Dwell Type_Rented"] + coef_mortgage * df_life["Dwell Type_With mortgage"] + stats.beta.rvs(a=2,b=2,loc=-0.5,size=df_life.shape[0])).clip(1, 5)
df_life['VALUE_WORK'] = np.round(value_work)

'''
Income equality
'''
# (Rate how much you value towards individual effort)
# Lower means incomes should make equal
df_life['ECONOMIC_VALUE_VALUE_IND_EFFORT'] = np.round((stats.beta.rvs(a=3.2,b=2,size=df_life.shape[0]) * 10).clip(1, 5))

'''
Private/ public
'''
# Lower means prefer private entities/ institutions
a = 2
b = 3.2
df_life['ECONOMIC_VALUE_VALUE_PRIVATE_PUBLIC_ENTITIES'] = np.round((stats.beta.rvs(a=a,b=b,size=df_life.shape[0]) * 10).clip(1, 5))

'''
Competition
'''
# Higher means good
a = 2
b = 4
df_life['ECONOMIC_VALUE_VALUE_COMPETITION'] = np.round((stats.beta.rvs(a=a,b=b,size=df_life.shape[0]) * 10).clip(1, 5))

'''
Security
'''
# Higher means feeling secured






