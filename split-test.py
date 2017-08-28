import numpy as np
import pandas as pd

df = pd.read_csv('/home/ahmed/Pictures/cogedis/24072017/maj+min/all_maj_min.csv',sep=',')

def train_validate_test_split(df, train_percent=.6, validate_percent=.2, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.ix[perm[:train_end]]
    validate = df.ix[perm[train_end:validate_end]]
    test = df.ix[perm[validate_end:]]
    return train, validate, test


if __name__ == '__main__':
    train,valid,test=train_validate_test_split(df,)
    train.to_csv('/home/ahmed/Pictures/cogedis/24072017/maj+min/all_maj_min_train.csv',sep=',',index=False)
    valid.to_csv('/home/ahmed/Pictures/cogedis/24072017/maj+min/all_maj_min_valid.csv', sep=',',index=False)
    test.to_csv('/home/ahmed/Pictures/cogedis/24072017/maj+min/all_maj_min_test.csv',sep=',',index=False)