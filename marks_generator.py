from random import randint
from random import choice
from random import choices
from random import gauss

import string

import numpy as np
import pandas as pd

def unikey_gen ():
    '''
    Generate unikey in the format of 4 letters and 4 numbers (e.g. 'cyan1001').
    '''
    unikey = ''.join(choice(string.ascii_lowercase) for _ in range(4))+'{:04d}'.format(randint(0,9999))
    return unikey

def exam_mark_gen (total, mu = 1, s = 0):
    possible_marks = np.array(range(total+1))   # Setting range as total + 1 means full marks can be awarded.
    weights = np.ones(total, dtype=np.int64)
    for x in np.nditer(weights):
        weights = np.multiply(weights,gauss(mu,s))
    print(weights)
    weights = weights.astype(int)
    print(weights)
    print()
    mark = choices(possible_marks)[0]
    print(mark)
    print()
    return mark

def data_gen (N):
    data = []
    for i in range(N):
        data.append([])
        data[i].append(unikey_gen())
        data[i].append(exam_mark_gen(20,15,15))
        data[i].append(exam_mark_gen(20,11,2))
        data[i].append(exam_mark_gen(15,5,3))
        data[i].append(exam_mark_gen(20,7,8))
        data[i].append(exam_mark_gen(15,8,15))
        data[i].append(exam_mark_gen(10,1,5))
    return data

N = 1
marks_df = pd.DataFrame([[unikey_gen(),1,1,1,1,1,1,2,2,2,2]])
marks_df.columns = ['unikey', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Quizes', 'Lab', 'Asgn 01', 'Asgn 02']
# print(marks_df.head())
print(data_gen(N))
