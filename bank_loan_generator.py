import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import math


def standardise_array(arr):
    return (arr - arr.mean())/(arr.std())

'''
Features: 

    Age - int 
    Gender - int, 1: Male, 0: Female
    Salary - int 
    Work experience - int
    Mortgage - int
    Fraudulent - int, 1: True, 0: False
'''

# N = 10
N = 1200

# Data generation
age_data = stats.norm.rvs(loc=26, scale=5,size=N)
age_data = np.array([int(x) for x in age_data])     # Cast to int
age_data_x = standardise_array(age_data)

gender_data = np.array(random.choices([1, 0], weights=[6,4], k=N))
gender_data_x = standardise_array(gender_data)

salary_data = np.floor(stats.pareto.rvs(2.66, size=N))*10000
salary_data_x = standardise_array(salary_data)

experience_data = stats.poisson.rvs(mu=[random.randint(2,5) for i in range(N)], size=N)
experience_data_x = standardise_array(experience_data)

qualification_data = np.array(random.choices(np.arange(1, 6), weights=[1, 3, 8, 5, 3], k=N))
qualification_data_x = standardise_array(qualification_data)

mortgage_data = np.floor(stats.poisson.rvs(mu=[random.randint(2,10) for i in range(N)], size=N))*1000000
mortgage_data_x = standardise_array(mortgage_data)

'''
Sample target variable generation 
'''
intercept = -3.22
b_age = 0.52
b_gender = 0.001
b_salary = 0.5
b_exp = -0.001
b_qual = 3.225
b_m = 8.95

y = np.array(1/(1 + np.exp(-(b_age*age_data_x + b_gender*gender_data_x + b_salary*salary_data_x + b_exp*experience_data_x + b_qual*qualification_data_x + b_m*mortgage_data_x))))

y[y > 0.5] = 1
y[y <= 0.5] = 0

# Export data

data = pd.DataFrame({
    'Fraud': y,
    'Age': age_data,
    'Gender': gender_data,
    'Salary': salary_data,
    'Work_experience': experience_data,
    'Qualification': qualification_data,
    'Mortgage': mortgage_data
})

data.to_csv('files/bank_loan_data.csv', index=False)