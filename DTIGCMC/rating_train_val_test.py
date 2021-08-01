import numpy as np
import torch
from utils import *
import random
from sklearn.model_selection import KFold

random_seed = 1234
kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
train_index = []
val_index = []
index = []
for i in range(0, samples_df.shape[0]):
    index.append(i)
for i, j in kf.split(index):
    train_index.append(list(i))
    val_index.append(list(j))

for k in range(0, 5):
    rating_train = []
    rating_val = []
    # print('-------------------------------------------------------------------------------------------------------------')
    for m in range(0, len(train_index[k])):
        rating_train.append(list(samples_df.values[train_index[k][m], :]))
    for n in range(0, len(val_index[k])):
        rating_val.append(list(samples_df.values[val_index[k][n], :]))

    rating_val = pd.DataFrame(rating_val)
    rating_train = pd.DataFrame(rating_train)
    rating_test = rating_val
    rating_cnt = 2
    pk = []
    for i, ratings in enumerate([rating_train, rating_val, rating_test]):
        rating_mtx_e = torch.zeros(size=(rating_cnt, num_protein_e, num_drug_e))

        for index, row in ratings.iterrows():
            u = row[0]
            v = row[1]
            r = row[2]

            rating_mtx_e[r, u, v] = 1
        pk.append(rating_mtx_e)
    torch.save(pk, './data/pk/pk_e_%d.pkl ' % k)