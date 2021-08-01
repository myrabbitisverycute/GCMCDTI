import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torchvision import models
import sklearn
import random

from layers import *
from metrics import rmse, softmax_accuracy, softmax_cross_entropy
from mxnet.gluon import loss as gloss
from sklearn import metrics

class GAE(nn.Module):
    def __init__(self, num_proteins, num_drugs, num_classes, num_side_features, nb,
                       p_features, d_features, p_features_side, d_features_side,
                       input_dim, emb_dim, hidden, dropout, slope, **kwargs):
        super(GAE, self).__init__()

        self.num_proteins = num_proteins
        self.num_drugs = num_drugs
        self.dropout = dropout
        self.slope = slope
        self.p_features = p_features
        self.d_features = d_features
        self.p_features_side = p_features_side
        self.d_features_side = d_features_side

        self.gcl1 = GraphConvolution(input_dim, hidden[0],
                                    num_proteins, num_drugs,
                                    num_classes, slope, self.dropout, bias=True)
        self.gcl2 = GraphConvolution(hidden[0], hidden[1],
                                    num_proteins, num_drugs,
                                    num_classes, slope, self.dropout, bias=True)
        self.denseu1 = nn.Linear(num_side_features, emb_dim, bias=True)
        self.densev1 = nn.Linear(num_side_features, emb_dim, bias=True)
        self.denseu2 = nn.Linear(emb_dim + hidden[1], hidden[2], bias=False)
        self.densev2 = nn.Linear(emb_dim + hidden[1], hidden[2], bias=False)

        self.bilin_dec = BilinearMixture(num_proteins=num_proteins, num_drugs=num_drugs,
                                         num_classes=num_classes,
                                         input_dim=hidden[2],
                                         nb=nb, dropout=0.)

    def forward(self, p, d, r_matrix):
        p_z, d_z = self.gcl1(self.p_features, self.d_features,
                             range(self.num_proteins), range(self.num_drugs), r_matrix)
        p_z, d_z = self.gcl2(p_z, d_z, p, d, r_matrix)
        p_f = torch.relu(self.denseu1(self.p_features_side[p]))
        d_f = torch.relu(self.densev1(self.d_features_side[d]))
        p_h = self.denseu2(F.dropout(torch.cat((p_z, p_f), 1), self.dropout))
        d_h = self.densev2(F.dropout(torch.cat((d_z, d_f), 1), self.dropout))
        output, m_hat= self.bilin_dec(p_h, d_h, p, d)
        r_mx = r_matrix.index_select(1, p).index_select(2, d)
        loss = softmax_cross_entropy(output, r_mx.float())
        rmse_loss = rmse(m_hat, r_mx.float())
        r_m = r_matrix.detach().numpy()

        row = []
        col = []
        label = []
        for i in range(0, r_m.shape[0]):
            for j in range(0, r_m.shape[1]):
                for k in range(0, r_m.shape[2]):
                    if r_m[i][j][k] == 1:
                        row.append(j)
                        col.append(k)
                        label.append(i)
                    else:
                        continue

        score = m_hat.detach().numpy()
        final_label = []
        for m in range(0, len(row)):
            final_label.append(score[row[m]][col[m]])

        data = []
        data.append(label)
        data.append(final_label)
        data = np.array(data).transpose()
        data = pd.DataFrame(data)
        data = data.values.tolist()
        random.seed(1234)
        random.shuffle(data)
        data = np.array(data)
        labels = list(data[:, 0])
        final_labels = list(data[:, 1])

        auc = metrics.roc_auc_score(labels, final_labels)
        results_val = [0 if j < 0.5 else 1 for j in final_labels]
        accuracy = metrics.accuracy_score(labels, results_val)
        pre = metrics.precision_score(labels, results_val)
        recall = metrics.recall_score(labels, results_val)
        mcc = metrics.matthews_corrcoef(labels, results_val)
        f1 = metrics.f1_score(labels, results_val)
        fpr, tpr, thresholds = metrics.roc_curve(labels, final_labels)
        tn, fp, fn, tp = metrics.confusion_matrix(labels, results_val).ravel()
        spec = tn / (tn + fp)
        return auc, loss, rmse_loss, accuracy, pre, recall, mcc, f1, fpr, tpr, spec
