import pandas as pd
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
args = parser.parse_args()
FILE_TYPE = args.type

if FILE_TYPE == 'binary':
    FILE_TYPE = 'binary_'
    LABEL = 'money_num'
else:
    FILE_TYPE = ''
    LABEL = 'label_num'


data = pd.read_pickle('../../data/'+ FILE_TYPE +'description.pkl')
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
for c in list(set(train[LABEL])):
    print(str(c) + ': ' + str(list(train[LABEL]).count(c)))
print("===")

### make test data description file
test = data.iloc[NUM_TRAIN:]
for c in list(set(test[LABEL])):
    print(str(c) + ': ' + str(list(test[LABEL]).count(c)))

print("\nPress the Enter key.")
input()
train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

### save description file
train.to_pickle('../../data/' + FILE_TYPE + 'train_description.pkl')
test.to_pickle('../../data/' + FILE_TYPE + 'test_description.pkl')