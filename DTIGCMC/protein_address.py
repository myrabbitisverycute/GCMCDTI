import pandas as pd
import numpy as np
import torch
from torch.nn.parameter import Parameter
import csv

proteinbase_header = ['protein_id', 'sequence']
proteinbase_df =pd.read_csv('data/DPdata/all sequence data.csv', names=proteinbase_header, engine='python', error_bad_lines=False)
proteinbase = np.array(proteinbase_df)
protein_id = proteinbase[:, 0]
protein_sequence = list(proteinbase[:, 1])
number_protein = len(protein_sequence)
e_header = ['id', 'drug']
e_df = pd.read_csv('data/DPdata/bind_orfhsa_drug_e2.csv', names=e_header, engine='python', error_bad_lines=False)
e_array = np.array(e_df)
e_id =list(np.unique(e_array[:, 0]))

for i in range(0,len(e_id)):
    if e_id[i] not in protein_id:
        print(e_id[i])

protein_sequence_e = []
for index, row in proteinbase_df.iterrows():
    if row['protein_id'] in e_id:
        protein_sequence_e.append(row['sequence'])

p_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19}

sq_list =[]
all_sq = []
for i in range(0, len(protein_sequence_e)):
    for j in range(0, len(protein_sequence_e[i])):
        if j == 1200:
            break
        if protein_sequence_e[i][j] in p_dict.keys():
            sq_list.append(p_dict[protein_sequence_e[i][j]])
    while len(sq_list)<1200:
        sq_list.append(0)
    all_sq.append(sq_list)
    sq_list = []

proteinsq_feature = []
for index in range(0, len(all_sq)):
    p_sequence = np.zeros((1200, 20), dtype=np.float64)
    p_sequence_row = 0
    for p in all_sq[index]:
        p_sequence[p_sequence_row, p] = 1.
        p_sequence_row = p_sequence_row +1
    p_sequence = p_sequence.transpose()  # [20,500]
    weight = Parameter(torch.randn(1200, 1)).to(torch.float64) # [500,1]
    p_sequence = torch.from_numpy(p_sequence)
    p_sequence = torch.mm(p_sequence, weight)  # [20,500]*[500,1] = [20,1]
    p_sequence = p_sequence.detach().numpy()  # [1,20]
    p_sequence = p_sequence.transpose()  # [1,20]
    p_sequence = np.squeeze(p_sequence)
    proteinsq_feature.append(p_sequence)
proteinsq_feature = np.array(proteinsq_feature)
protein_feat = proteinsq_feature  # [664,20]
