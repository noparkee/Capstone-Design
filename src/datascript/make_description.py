import pickle
import pandas as pd
import numpy as np
import os
import re

PATH = '../../data/'

FOLDER = os.listdir(PATH)

path_lst = []
name_lst = []
label_lst = []
label_num_lst = []

for i, F in enumerate(FOLDER):
    if F == 'data.zip' or F == 'description.pkl':
        continue
        
    path = PATH + F + '/'
    FILE = os.listdir(path)
    
    name_lst += FILE
    path_lst += list(map(lambda x: path[3:]+x, FILE))
    label_lst += [F for _ in range(len(FILE))]
    label_num_lst += [i for _ in range(len(FILE))]
    
# make dataframe
data = pd.DataFrame({'path': path_lst, 'name': name_lst, 'label': label_lst, 'label_num': label_num_lst})

# shuffle
data = data.sample(frac=1).reset_index(drop=True)

# save to pickle
data.to_pickle('../../data/description.pkl')
