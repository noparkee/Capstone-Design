import pandas as pd
import os

PATH = '../../data/'

path_lst = []
name_lst = []
label_lst = []
label_num_lst = []
shape_lst = []     # 동전 - 0, 지폐 - 1

FOLDER = ['10won', '100won', '1000won', '10000won', '50won', '500won', '5000won', '50000won', 'others']

for i, F in enumerate(FOLDER):
    if F == 'data.zip' or F == 'description.pkl':
        continue
        
    path = PATH + F + '/'
    FILE = os.listdir(path)
    
    name_lst += FILE
    path_lst += list(map(lambda x: path[3:]+x, FILE))
    label_lst += [F for _ in range(len(FILE))]
    label_num_lst += [i for _ in range(len(FILE))]
    
    #if F in ['10won', '100won', '50won', '500won']:     # 동전
    #    shape_lst += [0 for _ in range(len(FILE))]
    #else:                                               # 지폐
    #    shape_lst += [1 for _ in range(len(FILE))]
    if F == 'others':                                   # 화폐 아닌 것들
        shape_lst += [0 for _ in range(len(FILE))]
    else:
        shape_lst += [1 for _ in range(len(FILE))]

# make dataframe
data = pd.DataFrame({'path': path_lst, 'name': name_lst, 'label': label_lst, 'label_num': label_num_lst, 'money_num': shape_lst})

# shuffle
data = data.sample(frac=1).reset_index(drop=True)

# save to pickle
data.to_pickle('../../data/description.pkl')
