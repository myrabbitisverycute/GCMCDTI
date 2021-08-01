import pandas as pd
import numpy as np
import csv
from drug_address import *
from protein_address import *
import random

for header in drug_e_id:
    d = dict()
    feats = drug_e_id
    d.update({f: i for i, f in enumerate(feats)})
num_drug = len(d)

for header in e_id:
    p = dict()
    feats = e_id
    p.update({f: i for i, f in enumerate(feats)})
num_protein = len(p)
"""
   读取drug与酶之间的关联文件，转化为数字表示
"""
e_header = ['e_id', 'drug_id']
e_df = pd.read_csv('data/DPdata/bind_orfhsa_drug_e2.csv', names=e_header, header=None, engine='python', error_bad_lines=False)
num_protein_e = len(np.unique(e_df.values[:, 0]))
num_drug_e = len(np.unique(e_df.values[:, 1]))
rating_mx_e = np.zeros((e_df.shape[0], 2), dtype=np.int64)
for _, row in e_df.iterrows():
    e_name = row['e_id'].strip()
    e_index = p[e_name]
    rating_mx_e[_, 0] = e_index

    d_name = row['drug_id'].strip()
    d_index = d[d_name]
    rating_mx_e[_, 1] = d_index

drug_protein = pd.DataFrame(rating_mx_e)
with open('data/DPdata/pos_sample_e.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for _, row in drug_protein.iterrows():
        writer.writerow([row[0], row[1]])

rating_e = np.zeros((num_protein_e, num_drug_e), dtype=np.int64)
for _, row in drug_protein.iterrows():
    rating_e[row[0], row[1]] = 1

row = []
col = []

for i in range(0, num_protein_e):
    for j in range(0, num_drug_e):
        if rating_e[i, j] ==0:
            row.append(i)
            col.append(j)

neg_sample_e = []
neg_sample_e.append(row)
neg_sample_e.append(col)
neg_sample_e= np.array(neg_sample_e)
neg_sample_e = neg_sample_e.transpose()
neg_sample_e = pd.DataFrame(neg_sample_e)

with open('data/DPdata/neg_sample_e.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    for _, row in neg_sample_e.iterrows():
        writer.writerow([row[0], row[1]])

drug_protein.columns = ['protein', 'drug']
drug_protein['label'] = 1

neg_sample_e.columns = ['protein', 'drug']
neg_sample_e['label'] = 0

random_negative = neg_sample_e.sample(n=drug_protein.shape[0], random_state=1234, axis=0)
samples_df = drug_protein.append(random_negative)

seed = 1234
data_array = samples_df.values.tolist()
random.seed(seed)
random.shuffle(data_array)
samples_df = pd.DataFrame(data_array)
