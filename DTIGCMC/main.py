import os
import numpy as np
import pandas as pd
from random import sample
import matplotlib.pyplot as plt
from scipy import interp
import warnings
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import BatchSampler, SequentialSampler


from config import *
from data_load import get_loader
from model import *
args = get_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_proteins, num_drugs, num_classes, num_side_features, num_features,\
p_features, d_features, p_features_side, d_features_side= get_loader(args.data_type)

p_features = torch.from_numpy(p_features).to(device).float()
d_features = torch.from_numpy(d_features).to(device).float()
p_features_side = torch.from_numpy(p_features_side).to(device)
d_features_side = torch.from_numpy(d_features_side).to(device)

epochs = 100

def train(): # 训练部分

    for i in range(0, 5):
        print('---------------------------------------------------------------------------------------------------------------------')
        print(' Fold %d  '% (i + 1))
        f = torch.load('data/pk/pk_e_%d.pkl' % i)
        rating_train = f[0].to(device)
        rating_val = f[1].to(device)
        rating_test = f[2].to(device)

        model = GAE(num_proteins, num_drugs, num_classes,
                    num_side_features, args.nb,
                    p_features, d_features, p_features_side, d_features_side,
                    num_proteins + num_drugs, args.emb_dim, args.hidden, args.dropout, args.slope)
        if torch.cuda.is_available():
            model.cuda()
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()

        model.apply(weight_reset)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=[args.beta1, args.beta2])
        for epoch in range(args.start_epoch, args.num_epochs):  # 从0~5
            model.train()

            train_loss = 0.
            train_rmse = 0.
            for s, u in enumerate(BatchSampler(SequentialSampler(sample(range(num_proteins), num_proteins)),
                                  batch_size=num_proteins, drop_last=False)):

                u = torch.from_numpy(np.array(u)).to(device).long()

                for t, v in enumerate(BatchSampler(SequentialSampler(sample(range(num_drugs), num_drugs)),
                                      batch_size=num_drugs, drop_last=False)):
                    v = torch.from_numpy(np.array(v)).to(device).long()
                    if len(torch.nonzero(torch.index_select(torch.index_select(rating_train, 1, u), 2, v))) == 0:
                        continue

                    auc, loss_ce, loss_rmse, accuracy, pre, recall, mcc, f1, fpr, tpr, spec= model(u, v, rating_train)

                    optimizer.zero_grad()
                    loss_ce.backward()
                    optimizer.step()

                    train_loss += loss_ce.item()
                    train_rmse += loss_rmse.item()

            log = 'epoch: '+str(epoch+1)+' loss_ce: %.4f'% train_loss \
                                        +' loss_rmse: %.4f '% train_rmse+ ' auc: %.4f '% auc

        model.eval()
        with torch.no_grad():
            u = torch.from_numpy(np.array(range(num_proteins))).to(device).long()
            v = torch.from_numpy(np.array(range(num_drugs))).to(device).long()
            auc, test_loss, test_rmse, accuracy, pre, recall, mcc, f1, fpr, tpr, spec = model(u, v, rating_test)
            test_auc = metrics.auc(fpr, tpr)

        print('[test loss] : %.4f' % test_loss.item() + ' [test rmse] : %.4f' % test_rmse.item() + ' test auc: %.4f ' % test_auc.item()+ 'acc: %.4f' % accuracy + ' pre: %.4f' % pre+ ' recall: %.4f'% recall+' mcc: %.4f'% mcc+ ' f1:%.4f'% f1 +' spec: %.4f'% spec)

    print('## Training Finished !')
    print('-----------------------------------------------------------------------------------------------')
    return auc_result, acc_result, pre_result, recall_result, mcc_result, f1_result, fprs, tprs, specs


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    if args.mode == 'train':
        auc, acc, pre, recall, mcc, f1, fprs, tprs, specs = train()
    mean_fpr = np.linspace(0, 1, 10000)
    tpr = []

    for i in range(len(fprs)):
        tpr.append(interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.4, label='ROC fold %d (AUC = %.4f)' % (i + 1, auc[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    auc_std = np.std(auc)
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, auc_std))

    std_tpr = np.std(tpr, axis=0)
    tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
    tpr_lower = np.maximum(mean_tpr - std_tpr, 0)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.savefig('data/validation.jpg', dpi=1200, bbox_inches='tight')
    plt.show()