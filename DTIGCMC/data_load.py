import os
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from torch.utils import data

from utils import *

def get_loader(data_type):
	SYM = True
	DATASET = data_type
	datasplit_path = 'data/' + DATASET + '/withfeatures.pickle'
	p_features, d_features, class_values = create_f(data_type, datasplit_path=datasplit_path)
	num_proteins = p_features.shape[0]
	num_drugs = d_features.shape[0]
	print("Normalizing feature vectors...")
	p_features_side = normalize_features(p_features)
	d_features_side = normalize_features(d_features)

	p_features_side, d_features_side = preprocess_drug_protein_features(p_features_side, d_features_side)
	p_features_side = np.array(p_features_side.todense(), dtype=np.float32)
	d_features_side = np.array(d_features_side.todense(), dtype=np.float32)

	num_side_features = p_features_side.shape[1]
	id_csr_p = sp.identity(num_proteins, format='csr')
	id_csr_d = sp.identity(num_drugs, format='csr')

	p_features, d_features = preprocess_drug_protein_features(id_csr_p, id_csr_d)

	p_features = p_features.toarray()
	d_features = d_features.toarray()

	num_features = p_features.shape[1]

	return num_proteins, num_drugs, len(class_values), num_side_features, num_features, \
		   p_features, d_features, p_features_side, d_features_side