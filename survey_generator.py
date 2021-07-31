import pandas as pd
import numpy as np
import scipy.stats as stats
import random

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

gender_data = random.choices([1, 0], weights=[7,3], k=N)

experience_data = stats.poisson.rvs(mu=[random.randint(2,5) for i in range(N)], size=N)
experience_data[(20+experience_data-age_data) < 0] = random.randint(1,2)
# print(experience_data[(20+experience_data-age_data) < 0])

qualifications = ['High school', 'Diploma/ Advanced Diploma', 'Bachelor', 'Masters', 'PhD']
qualification_data = random.choices(qualifications, weights=[1, 3, 8, 5, 3], k=N)

role = ['Software Engineer', 'Business Analyst', 'Data Analyst', 'Data Engineer', 'Data Scientist', 'Database Engineer', 'Statistician', 'Unemployed', 'Other']
role_data = random.choices(role, weights=[6, 3, 4, 2, 3, 2, 2, 5, 0.5], k=N)

lang = ['Python', 'R', 'PHP', 'C']
lang_data = random.choices(lang, weights=[6, 2, 1, 1], k=N)

salary_data = np.floor(stats.pareto.rvs(2.66, size=N))*10000

data = pd.DataFrame({'Age': age_data,
                     'Gender': gender_data,
                     'Qualifications': qualification_data,
                     'Role': role_data,
                     'Programming_language': lang_data,
                     'Experience': experience_data,
                     'Salary': salary_data
                     })

data.to_csv('files/survey_data.csv', index=False)