import pandas as pd
import numpy as np

drug = pd.read_csv('data/DPdata/all drug fingerprints.csv', header=None, engine='python', error_bad_lines=False)
drug_id = list(drug.values[:, 0])
for i in range(0, len(drug_id)):
    drug_id[i].strip()
drug_feat = drug.values[:, 1:].astype(dtype=np.float32)
drug_feat_e = []
drug_e_header = ['e_id', 'drug_id']
drug_e_df = pd.read_csv('data/DPdata/bind_orfhsa_drug_e2.csv', names=drug_e_header, engine='python', error_bad_lines=False)

drug_e_id = list(np.unique(drug_e_df.values[:, 1]))
for i in range(0, len(drug_e_id)):
    drug_e_id[i].lstrip()

for i in range(0, len(drug_id)):
    if drug_id[i] in drug_e_id:
        drug_feat_e.append(list(drug.values[i, 1:]))
    else:
        continue

drug_feat_e = np.array(drug_feat_e)
count = 0
row = []
col = []
for i in range(0, drug_feat_e.shape[0]):
    for j in range(0, 615):
        if drug_feat_e[i][j] != drug_feat_e[i][j]:
            drug_feat_e[i][j]=0
            row.append(i)
            col.append(j)
            count = count +1

cnt =0
for i in range(0, drug_feat_e.shape[0]):
    for j in range(0, 615):
        if drug_feat_e[i][j] != drug_feat_e[i][j]:
            row.append(i)
            col.append(j)
            cnt = cnt +1

drug_drug_feat = drug_feat_e  # [445,615]