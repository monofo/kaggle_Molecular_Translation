import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from configure import *

def seq_cross_entropy_loss(logit, token, length):
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    return loss

# https://www.aclweb.org/anthology/2020.findings-emnlp.276.pdf
def seq_anti_focal_cross_entropy_loss(logit, token, length):
    gamma = 0.5 # {0.5,1.0}
    label_smooth = 0.90

    #---
    truth = token[:, 1:]
    L = [l - 1 for l in length]
    logit = pack_padded_sequence(logit, L, batch_first=True).data
    truth = pack_padded_sequence(truth, L, batch_first=True).data
    #loss = F.cross_entropy(logit, truth, ignore_index=STOI['<pad>'])
    #non_pad = torch.where(truth != STOI['<pad>'])[0]  # & (t!=STOI['<sos>'])


    # ---
    #p = F.softmax(logit,-1)
    #logp = - torch.log(torch.clamp(p, 1e-4, 1 - 1e-4))

    logp = F.log_softmax(logit, -1)
    logp = logp.gather(1, truth.reshape(-1,1)).reshape(-1)
    p = logp.exp()

    #loss = - ((1 - p) ** gamma)*logp  #focal
    loss = - ((1 + p) ** gamma)*logp  #anti-focal
    loss = loss.mean()
    return loss


def np_loss_cross_entropy(probability, truth):
    batch_size = len(probability)
    truth = truth.reshape(-1)
    p = probability[np.arange(batch_size),truth]
    loss = -np.log(np.clip(p,1e-6,1))
    loss = loss.mean()
    return loss



def get_criterion(criterion_name, params):
    if criterion_name == "cross_entropy":
        criterion = seq_cross_entropy_loss
    elif criterion_name == "anti_focal":
        criterion = seq_anti_focal_cross_entropy_loss
    
    return criterion