from __future__ import division
from __future__ import print_function

import os
import random
from io import *
import shutil

import pandas as pd
import numpy as np
import scipy.sparse as sp
import pickle as pkl
from urllib.request import urlopen
from zipfile import *
from drug_address import *
from protein_address import *
from drug_protein_rating import *
from mxnet import ndarray as nd, gluon, autograd
import dgl
from config import *
import torch.nn as nn

def normalize_features(feat):

    degree = np.asarray(feat.sum(1)).flatten()
    degree[degree == 0.] = np.inf
    degree_inv = 1. / degree
    degree_inv_mat = sp.diags([degree_inv], [0])
    feat_norm = degree_inv_mat.dot(feat)
    if feat_norm.nnz == 0:
       print('ERROR: normalized adjacency matrix has only zero entries')
       exit

    return feat_norm

def weight_reset(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

def preprocess_drug_protein_features(p_features, d_features):
    zero_csr_p = sp.csr_matrix((p_features.shape[0], d_features.shape[1]), dtype=p_features.dtype)
    zero_csr_d = sp.csr_matrix((d_features.shape[0], p_features.shape[1]), dtype=d_features.dtype)
    p_features = sp.hstack([p_features, zero_csr_p], format='csr')
    d_features = sp.hstack([zero_csr_d, d_features], format='csr')
    return p_features, d_features

def map_data(data):
    uniq = list(set(data))
    id_dict = {old: new for new, old in enumerate(sorted(uniq))}
    data = np.array(list(map(lambda x: id_dict[x], data)))
    n = len(uniq)
    return data, id_dict, n

def load_data(verbose=True):
    d_features = drug_drug_feat
    p_features = protein_feat
    d_node_rating = samples_df.values[:, 1]
    p_node_rating = samples_df.values[:, 0]
    ratings = samples_df.values[:, 2]

    d_features = sp.csr_matrix(d_features)
    p_features = sp.csr_matrix(p_features)

    if verbose:
        print('Number of drugs = %d' % num_drug)
        print('Number of proteins = %d' % num_protein)
        print('Number of links = %d' % ratings.shape[0])
        print('Fraction of positive links = %.4f' % (float(ratings.shape[0]) / (num_drug * num_protein)))

    return num_protein, num_drug, ratings, p_node_rating, d_node_rating, p_features, d_features

def create_f(dataset, seed=1234, datasplit_path=None, verbose=True):
    num_protein, num_drug, ratings, p_node, d_node, p_features, d_features = load_data(verbose=verbose)

    with open(datasplit_path, 'wb') as f:
        pkl.dump([num_protein, num_drug, ratings, p_node, d_node, p_features, d_features], f)

    class_values = np.array([0, 1])

    return p_features, d_features, class_values

