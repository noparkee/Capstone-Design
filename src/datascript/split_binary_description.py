import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, required=True)
args = parser.parse_args()
FILE = args.file



data = pd.read_pickle('../../data/binary_description.pkl')
#data = data.sample(frac=1).reset_index(drop=True)
print(data)

blur = data[data['name'].str.contains('blur')]
data = data.drop(blur.index).reset_index(drop=True)
print("after drop blur")
print(data)

TOTAL = len(data)
NUM_TRAIN = int(TOTAL * 0.7)        # train : test = 7 : 3
NUM_TEST = TOTAL - NUM_TRAIN

### make train data description file
train = data.iloc[:NUM_TRAIN]
train = pd.concat([train, blur])
train = train.sample(frac=1).reset_index(drop=True)
for c in list(set(train['money_num'])):
    print(str(c) + ': ' + str(list(train['money_num']).count(c)))
print("===")

### make test data description file
test = data.iloc[NUM_TRAIN:]
for c in list(set(test['money_num'])):
    print(str(c) + ': ' + str(list(test['money_num']).count(c)))

print("\nPress the Enter key.")
input()
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

train.to_pickle('../../data/binary_train_description.pkl')
test.to_pickle('../../data/binary_test_description.pkl')
