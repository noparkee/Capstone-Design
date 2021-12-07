import pandas as pd
import os

PATH = '../../data/'

path_lst = []
name_lst = []
label_lst = []
label_num_lst = []

FOLDER = ['10000won', '5000won', '10000won', '50000won']

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
data.to_pickle('../../data/paper_description.pkl')
