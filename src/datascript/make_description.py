import os
import argparse
import pandas as pd



parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
args = parser.parse_args()
FILE_TYPE = args.type


PATH = '../../data/'

path_lst = []
name_lst = []
label_lst = []
label_num_lst = []
shape_lst = []     # 동전 - 0, 지폐 - 1

if FILE_TYPE == 'binary':
    FOLDER = ['10won', '100won', '1000won', '10000won', '50won', '500won', '5000won', '50000won', 'others']
    FILE_NAME = 'binary_description.pkl'
else:
    FOLDER = ['10won', '100won', '1000won', '10000won', '50won', '500won', '5000won', '50000won']
    FILE_NAME = 'description.pkl'


for i, F in enumerate(FOLDER):        
    path = PATH + F + '/'
    FILE = os.listdir(path)
    
    name_lst += FILE
    path_lst += list(map(lambda x: path[3:]+x, FILE))
    label_lst += [F for _ in range(len(FILE))]
    label_num_lst += [i for _ in range(len(FILE))]
    
    if F == 'others':                                   # 화폐 아닌 것들
        shape_lst += [0 for _ in range(len(FILE))]
    else:
        shape_lst += [1 for _ in range(len(FILE))]


# make dataframe
data = pd.DataFrame({'path': path_lst, 'name': name_lst, 'label': label_lst, 'label_num': label_num_lst, 'money_num': shape_lst})

# shuffle
data = data.sample(frac=1).reset_index(drop=True)

# save to pickle
if FILE_TYPE == 'binary':
    data.to_pickle('../../data/binary_description.pkl')
else:
    data.to_pickle('../../data/description.pkl')
