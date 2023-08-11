import typing

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../../')
from losses import our_loss1, our_loss2


class MoCoLosses(typing.NamedTuple):
    loss1: typing.Optional[torch.Tensor] = None
    loss2: typing.Optional[torch.Tensor] = None

    def combine(self,  loss1_w: float=1, loss2_w: float=1) -> torch.Tensor:
        l = loss1_w*self.loss1+loss2_w*self.loss2
        return l


class MoCo(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999,
                 contr_tau=0.07, Ny=None, Ns=None, unif_intra_batch=True, mlp=False, tau_pos=1.0, tau_neg=1.0, alpha=1.0, beta=1.0, with_debiased=False, tau_plus=0.1):
        r"""
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        tau_pos, tau_neg: positive/negative temperature (default: 1.00)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m

        # define my added params
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.alpha = alpha
        self.beta = beta
        self.with_debiased = with_debiased
        self.tau_plus = tau_plus

        # positive samples 
        self.Ny = Ny

        # negative samples
        self.Ns = Ns
        self.unif_intra_batch = unif_intra_batch

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        r"""
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        r"""
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        r"""
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]
    
    def forward_negative_loss(self, z_x, z_y):
        '''
        pairwise cost function
        '''
        cost_for_l2 = torch.norm(z_x[:,None] - z_y, dim=-1).pow(2).mul(-self.beta)
        # mask the diagnal to select i != j as negative pairs
        cost_for_l2 = cost_for_l2.masked_select(~torch.from_numpy(np.eye(cost_for_l2.shape[0], dtype=bool)).cuda()).view(cost_for_l2.shape[0], cost_for_l2.shape[0]-1)
                    
        
        '''
        pairwise distance function
        '''
        dist_for_l2 = cost_for_l2.mul(1.0/self.beta)
        '''
        conditional distribution calculation
        '''
        weights_for_l2 = torch.softmax(dist_for_l2, dim=1)

        loss = (cost_for_l2*weights_for_l2).sum(1).mean()
        return loss, weights_for_l2

    def forward(self, im_list, epoch=1):
        r"""
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            MoCoLosses object containing the loss terms (and logits if contrastive loss is used)
        """
        # lazyily computed & cached!
        def get_q_bdot_k(q, k):
            if not hasattr(get_q_bdot_k, 'result'):
                get_q_bdot_k.result = (q * k).sum(dim=1)
            assert get_q_bdot_k.result._version == 0
            return get_q_bdot_k.result

        # lazyily computed & cached!
        def get_q_dot_queue(q):
            if not hasattr(get_q_dot_queue, 'result'):
                get_q_dot_queue.result = q @ self.queue.clone().detach()
            assert get_q_dot_queue.result._version == 0
            return get_q_dot_queue.result

        ############### update the key encoder ####################
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
        
        # Get the latent codes of all samples
        q_list = []
        k_list = []
        for x in range(len(im_list)):
            im = im_list[x]
            # compute query features
            q = self.encoder_q(im)  # queries: Mxd
            q = F.normalize(q, dim=1)
            q_list.append(q)

            # compute key features
            with torch.no_grad():  # no gradient to keys with momentum encoder
                # shuffle for making use of BN
                im_k, idx_unshuffle = self._batch_shuffle_ddp(im)
                k = self.encoder_k(im_k)  # keys: Mxd
                k = F.normalize(k, dim=1)
                # undo shuffle
                k = self._batch_unshuffle_ddp(k, idx_unshuffle)
            k_list.append(k)
        stacked_k = torch.stack(k_list, dim=0)
        
        ################### compute for CACR loss ####################
        moco_loss_ctor_dict = {}
        moco_loss_ctor_dict['loss1'] = 0.0
        moco_loss_ctor_dict['loss2'] = 0.0
        intra_loss2 = 0.0
        intra_counter = 0
        sq_loss2 = 0.0
        for x in range(len(im_list)):
            q = q_list[x]
            # generate mask
            mask = list(range(len(im_list)))
            mask.pop(x)

            ##################### compute for loss 1: attraction ##################### 
            '''
            cost function
            '''
            cost_for_l1 = (q - stacked_k[mask]).norm(p=2, dim=-1).pow(2).transpose(1, 0)

            with torch.no_grad():
                # calculation involves momentum encoder, so with no grad here.
                weights_for_l1 = torch.softmax(cost_for_l1.mul(self.tau_pos), dim=1)

            moco_loss_ctor_dict['loss1'] += (cost_for_l1*weights_for_l1).sum(1).mean().mul(self.alpha)
            ##################### end for loss 1 ##################### 

            #################### compute for loss 2 (repulsion) in terms of queue ######################
            # cost function (|x - y|^2 = 2 - 2 * xTy)
            sq_dists_for_cost = (2 - 2 * get_q_dot_queue(q))
            sq_dists_for_weights = sq_dists_for_cost.detach()
            if self.unif_intra_batch:
                intra_batch_sq_dists = torch.norm(q[:,None] - q, dim=-1).pow(2).masked_select(~torch.from_numpy(np.eye(q.shape[0], dtype=bool)).cuda()).view(q.shape[0], q.shape[0] - 1)
                # combine the distance of negative samples from the queue and intra-batch: Mx(K+M-1)
                sq_dists_for_cost = torch.cat([sq_dists_for_cost, intra_batch_sq_dists], dim=1) #
                sq_dists_for_weights = torch.cat([sq_dists_for_weights, intra_batch_sq_dists], dim=1)
                
            weights_for_l2 = torch.softmax(sq_dists_for_weights.mul(-self.tau_neg), dim=1) # compute the conditional distribution
            moco_loss_ctor_dict['loss2'] += (sq_dists_for_cost.mul(-1.0)*weights_for_l2).sum(1).mean().mul(self.beta) # elementwise product of cost and probability
            ##################### end for loss 2 #####################
        moco_loss_ctor_dict['loss1'] = moco_loss_ctor_dict['loss1']/len(im_list)
        moco_loss_ctor_dict['loss2'] = moco_loss_ctor_dict['loss2']/len(im_list)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_list[-1])

        # test code
        if (epoch+1)%50==0:
            print('#############Weights for Positive Loss###############')
            print(weights_for_l1.shape, weights_for_l1.data.cpu().numpy().max(1), weights_for_l1.data.cpu().numpy().min(1))

            print('#############Weights for Negative Loss###############')
            print(weights_for_l2.shape, weights_for_l2.data.cpu().numpy().max(1), weights_for_l2.data.cpu().numpy().min(1))

        return MoCoLosses(**moco_loss_ctor_dict)

    


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    r"""
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

