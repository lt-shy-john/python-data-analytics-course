import pandas as pd
import numpy as np
import scipy.stats as stats
import random

np.set_printoptions(suppress=True)

'''
Features: 

    Age - int 
    Gender - int, 1: Male, 0: Female
    Experience - int
    Formal qualifications - str 
    Role - str 
    Years learned programming - int 
    Programming language - str
    Salary - int 
'''

# N = 10
N = 1000

# Data generation
age_data = stats.norm.rvs(loc=26, scale=5,size=N)
age_data = np.array([int(x) for x in age_data])     # Cast to int
age_data[age_data <=19] = random.randint(25, 45)

gender_data = np.array(random.choices([1, 0], weights=[7,3], k=N))

experience_data = stats.poisson.rvs(mu=[random.randint(2,5) for i in range(N)], size=N)
experience_data[experience_data[(20+experience_data-age_data) < 0]] = random.randint(1,3)
# print(experience_data[(20+experience_data-age_data) < 0])

qualifications = ['High school', 'Diploma/ Advanced Diploma', 'Bachelor', 'Masters', 'PhD']
qualification_data = random.choices(qualifications, weights=[1, 3, 8, 5, 3], k=N)

role = ['Software Engineer', 'Business Analyst', 'Data Analyst', 'Data Engineer', 'Data Scientist', 'Database Engineer', 'Statistician', 'Unemployed', 'Other']
role_data = random.choices(role, weights=[6, 3, 4, 2, 3, 2, 2, 5, 0.5], k=N)
role_coef = [121.55, 100.54, 96.23, 105.11, 98.10, 115.13, 125.20, 0, 89.11]

f = lambda x: role.index(x)
role_salary_data = np.array([round((f(x) + 1) * role_coef[f(x)], 2) for x in role_data])

lang = ['Python', 'R', 'PHP', 'C', 'JavaScript', 'Java']
lang_data = random.choices(lang, weights=[5, 1, 1, 2, 1, 4], k=N)
lang_coef = [56.2, 34.2, 45.2, 65.10, 50.10, 49.15]

g = lambda x: lang.index(x)
lang_salary_data = np.array([round((g(x) + 1) * lang_coef[g(x)], 2) for x in lang_data])

salary_data = 165.22\
              - 0.25 * (age_data - 48)**2\
              + 42.33 * gender_data\
              + 1.33 * experience_data\
              + role_salary_data\
              + lang_salary_data\
              + np.floor(stats.pareto.rvs(1.66, size=N))
salary_data = salary_data * 2000
# print(salary_data)

data = pd.DataFrame({'Age': age_data,
                     'Gender': gender_data,
                     'Qualifications': qualification_data,
                     'Role': role_data,
                     'Programming_language': lang_data,
                     'Experience': experience_data,
                     'Salary': salary_data
                     })

# Correct salary data
data.loc[data['Role'] == 'Unemployed', 'Salary'] = 0

data.to_csv('files/survey_data.csv', index=False)