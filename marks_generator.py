from random import randint
from random import choice
from random import choices
from random import randrange

import string

import numpy as np
import pandas as pd

def unikey_gen():
    '''
    Generate unikey in the format of 4 letters and 4 numbers (e.g. 'cyan1001').
    '''
    unikey = ''.join(choice(string.ascii_lowercase) for _ in range(4))+'{:04d}'.format(randint(0,9999))
    return unikey

def exam_mark_gen(total):
    mark = randrange(0,2*(total+1))/2
    return mark

def asgn_mark_gen():
    mark = ''.join((str(randrange(100*100)/100),'%'))
    return mark

def data_gen(N):
    data = []
    for i in range(N):
        data.append([])
        data[i].append(unikey_gen())
        data[i].append(exam_mark_gen(20))
        data[i].append(exam_mark_gen(20))
        data[i].append(exam_mark_gen(15))
        data[i].append(exam_mark_gen(20))
        data[i].append(exam_mark_gen(15))
        data[i].append(exam_mark_gen(10))
        data[i].append(asgn_mark_gen())
        data[i].append(asgn_mark_gen())
        data[i].append(asgn_mark_gen())
        data[i].append(asgn_mark_gen())
    return data

N = 250
marks_df = pd.DataFrame(data_gen(N))
marks_df.columns = ['unikey', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Quizes', 'Lab', 'Asgn 01', 'Asgn 02']
# print(marks_df)
marks_df.to_csv('marks.csv', encoding='utf-8', index=False)
