import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from rating_train_val_test import *
from model import *

class GraphConvolution(Module):
    def __init__(self, input_dim, hidden_dim, num_proteins, num_drugs, num_classes, slope, dropout, bias=True):
        super(GraphConvolution, self).__init__()

        self.act = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout)
        self.p_weight = Parameter(torch.randn(num_classes, input_dim, hidden_dim))
        self.d_weight = self.p_weight
        if bias:
            self.p_bias = Parameter(torch.randn(hidden_dim))
            self.d_bias = self.p_bias
        else:
            self.p_bias = None
            self.d_bias = None

        for w in [self.p_weight, self.d_weight]:
            nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))

    def normalize(self, mx):
        rowsum = torch.sum(mx, 0)
        r_inv = torch.pow(rowsum, -0.5)
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        colsum = torch.sum(mx, 1)
        c_inv = torch.pow(colsum, -0.5)
        c_inv[torch.isinf(c_inv)] = 0.
        c_mat_inv = torch.diag(c_inv)
        mx = torch.matmul(mx, r_mat_inv)
        mx = torch.matmul(c_mat_inv, mx)
        return mx

    def forward(self, p_feat, d_feat, p, d, support):

        p_feat = self.dropout(p_feat)
        d_feat = self.dropout(d_feat)

        supports_p = []
        supports_d = []
        p_weight = 0
        d_weight = 0
        for r in range(support.size(0)):
            p_weight = p_weight + self.p_weight[r]
            d_weight = d_weight + self.d_weight[r]

            tmp_p = torch.mm(p_feat, p_weight)
            tmp_d = torch.mm(d_feat, d_weight)

            support_norm = self.normalize(support[r])
            support_norm_t = self.normalize(support[r].t())

            supports_p.append(torch.mm(support_norm[p], tmp_d))
            supports_d.append(torch.mm(support_norm_t[d], tmp_p))

        z_p = torch.sum(torch.stack(supports_p, 0), 0)
        z_d = torch.sum(torch.stack(supports_d, 0), 0)
        if self.p_bias is not None:
            z_p = z_p + self.p_bias
            z_d = z_d + self.d_bias

        p_outputs = self.act(z_p)
        d_outputs = self.act(z_d)
        return p_outputs, d_outputs

class BilinearMixture(Module):
    def __init__(self, num_proteins, num_drugs, num_classes, input_dim,
                 nb=1, dropout=0.7, **kwargs):
        super(BilinearMixture, self).__init__(**kwargs)

        self.dropout = nn.Dropout(dropout)
        self.weight = Parameter(torch.randn(nb, input_dim, input_dim))
        self.a = Parameter(torch.randn(nb, num_classes))

        self.u_bias = Parameter(torch.randn(num_proteins, num_classes))
        self.v_bias = Parameter(torch.randn(num_drugs, num_classes))

        for w in [self.weight, self.a, self.u_bias, self.v_bias]:
            nn.init.xavier_normal_(w)

    def forward(self, u_hidden, v_hidden, u, v):

        u_hidden = self.dropout(u_hidden)
        v_hidden = self.dropout(v_hidden)

        basis_outputs = []
        for weight in self.weight:
            u_w = torch.matmul(u_hidden, weight)
            x = torch.matmul(u_w, v_hidden.t())
            basis_outputs.append(x)

        basis_outputs = torch.stack(basis_outputs, 2)
        outputs = torch.matmul(basis_outputs, self.a)
        outputs = outputs + self.u_bias[u].unsqueeze(1).repeat(1,outputs.size(1), 1)\
                          + self.v_bias[v].unsqueeze(0).repeat(outputs.size(0), 1, 1)
        outputs = outputs.permute(2, 0, 1)
        softmax_out = F.softmax(outputs, 0)
        m_hat = torch.stack([r*output for r, output in enumerate(softmax_out)], 0)
        m_hat = torch.sum(m_hat, 0)

        return outputs, m_hat
