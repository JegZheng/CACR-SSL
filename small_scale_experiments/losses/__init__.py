'''
This script defines the loss functions for all kinds of methods
Implemented by Huangjie Zheng and Xu Chen
'''
import torch
import numpy as np
import torch.nn as nn

def our_loss1(latent_codes_list, alpha=1.0, tau_pos=1.0, choice='full'):
    # loss1: contrastive attraction
    loss1 = 0.0
    for x in range(len(latent_codes_list)): # iterate each query
        z_x = latent_codes_list[x]

        # Given the query, the other samples are negatives
        mask = list(range(len(latent_codes_list)))
        mask.pop(x)
        masked_z_x = torch.stack(latent_codes_list, dim=0)[mask]

        '''
        pairwise cost function
        '''
        cost = (z_x - masked_z_x).norm(p=2, dim=-1).pow(2).transpose(1, 0).mul(alpha)

        '''
        pairwise distance function
        '''
        dist = cost.mul(tau_pos/alpha)
        '''
        conditional distribution calculation
        '''
        weights = torch.softmax(dist, dim=1)

        if choice in ['none', 'without_w1']:
            loss1 += (cost).mean() # if without the positive conditional distribution, all sample pairs are uniformly transported
        else:
            loss1 += (cost * weights).sum(1).mean() # element-wise product of cost and transport probability

    return loss1 / (len(latent_codes_list)), weights

'''
Euclidean Cost Metric
'''
def forward_negative_loss(z_x, z_y, beta=1.0, tau_neg=1.0, tau_plus=0.1, choice='full', with_debiased=False):
    '''
    pairwise cost function
    '''
    cost = torch.norm(z_x[:,None] - z_y, dim=-1).pow(2).mul(-beta)
    # mask the diagnal to select i != j as negative pairs
    cost = cost.masked_select(~torch.from_numpy(np.eye(cost.shape[0], dtype=bool)).cuda()).view(cost.shape[0], cost.shape[0] - 1)
    
    '''
    pairwise distance function
    '''
    dist = cost.mul(tau_neg/beta)
    '''
    conditional distribution calculation
    '''
    weights = torch.softmax(dist, dim=1)
    
    if choice in ['none', 'without_w2']:
        loss = (cost).mean() # if without the negative conditional distribution, all sample pairs are uniformly transported
    else:
        if with_debiased: # if combine with debiasing methods
            N = z_x.shape[0] * 2 - 2
            pos = (z_x - z_y).norm(p=2, dim=-1).pow(2)
            reweight_neg = (cost * weights).sum(1)
            loss = ((-tau_plus*pos+reweight_neg)/(1-tau_plus)).mean()
        else:
            loss = (cost * weights).sum(1).mean() # element-wise product of cost and transport probability
    return loss, weights

# '''
# RBF Cost Metric
# To use this loss, please uncomment the following codes and comment the above codes
# '''
# def forward_negative_loss(z_x, z_y, beta=1.0, tau_neg=1.0, tau_plus=0.1, t=2.0, choice='full', with_debiased=False):
#     # t=2.0 on CIFAR10, CIFAR100 and STL10
#     '''
#     cost function
#     '''
#     cost = torch.norm(z_x[:,None] - z_y, dim=-1).pow(2).mul(-t).exp()
#     # have masked cost for final costs
#     cost = cost.masked_select(~torch.from_numpy(np.eye(cost.shape[0], dtype=bool)).cuda()).view(cost.shape[0],
#                                                                                                         cost.shape[0] - 1)
    
#     '''
#     distance function
#     '''
#     dist = cost.log().mul(tau_neg/t)
#     '''
#     conditional distribution calculation
#     '''
#     weights = torch.softmax(dist, dim=1)
    
#     if choice in ['none', 'without_w2']:
#         loss = (cost).mean().log().mul(beta)
#     else:
#         loss = (cost * weights).sum(1).mean().log().mul(beta)
#     return loss, weights

    
def our_loss2(latent_codes_list, beta=1.0, tau_neg=1.0, tau_plus=0.1, choice='full', with_debiased=False):
    # loss2: contrastive repulsion
    loss2 = 0.0
    counter = 0
    for x in range(len(latent_codes_list)):
        z_x = latent_codes_list[x]
        for y in range(x+1, len(latent_codes_list)):
            z_y = latent_codes_list[y]
            loss_for, weights= forward_negative_loss(z_x, z_y, beta=beta, tau_neg=tau_neg, tau_plus=tau_plus, choice=choice, with_debiased=with_debiased)
            loss2 += loss_for
            counter += 1
    return loss2 / counter, weights

__all__ = ['our_loss1', 'our_loss2']

