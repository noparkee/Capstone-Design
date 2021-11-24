import pandas as pd



TOTAL = 2245
NUM_TRAIN = int(TOTAL * 0.7)
NUM_TEST = TOTAL - NUM_TRAIN

data = pd.read_pickle('../../data/description.pkl')
#data = data.sample(frac=1).reset_index(drop=True)

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
train.to_pickle('../../data/train_description.pkl')
test.to_pickle('../../data/test_description.pkl')