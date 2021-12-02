import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()
FILE = args.file



data = pd.read_pickle('../../data/' + FILE + '.pkl')
#data = data.sample(frac=1).reset_index(drop=True)
TOTAL = len(data)
NUM_TRAIN = int(TOTAL * 0.8)        # train : test = 7 : 3
NUM_TEST = TOTAL - NUM_TRAIN

print(data)
train = data.iloc[:NUM_TRAIN]
for c in list(set(train['label_num'])):
    print(str(c) + ': ' + str(list(train['label_num']).count(c)))
print("===")
test = data.iloc[NUM_TRAIN:]
for c in list(set(test['label_num'])):
    print(str(c) + ': ' + str(list(test['label_num']).count(c)))
input()
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

if FILE == 'description':
    train.to_pickle('../../data/train_description.pkl')
    test.to_pickle('../../data/test_description.pkl')
elif FILE == 'coin_description':
    train.to_pickle('../../data/coin_train_description.pkl')
    test.to_pickle('../../data/coin_test_description.pkl')
elif FILE == 'paper_description':
    train.to_pickle('../../data/paper_train_description.pkl')
    test.to_pickle('../../data/paper_test_description.pkl')