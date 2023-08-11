# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def cacr_rbfloss(self, q_list, k):
        '''
        q_list: list of query embeddings
        k: momentum embedding
        '''
        # normalize
        qs = [nn.functional.normalize(i, dim=1) for i in q_list]
        k = nn.functional.normalize(k, dim=1)
        with torch.no_grad():
            p = torch.softmax(torch.stack([ ((i - k)**2).sum(-1) for i in qs])  / self.T, dim=0)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        sim = [torch.einsum('nc,mc->nm', [i, k]) / self.T for i in qs]
        N = sim[0].shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        loss = 0
        for i in range(len(sim)):
            loss += (nn.CrossEntropyLoss(reduction='none')(sim[i], labels) * p[i]).mean()  * (2 * self.T)
        return loss


    def cacr_euclideanloss(self, q_list, k):
        '''
        q_list: list of query embeddings
        k: momentum embedding
        '''
        # normalize
        qs = [nn.functional.normalize(i, dim=1) for i in q_list]
        k = nn.functional.normalize(k, dim=1)
        with torch.no_grad():
            p = torch.softmax(torch.stack([ ((i - k)**2).sum(-1) for i in qs])  / self.T, dim=0)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive 
        cost = [1 - torch.einsum('nc,mc->nm', [i, k]) for i in qs]
        N = cost[0].shape[0]  # batch size per GPU
        M = cost[0].shape[1]  # global batch size
        loss = 0
        for i in range(len(cost)):
            ca = torch.diag(cost[i][:, torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()])
            mask = torch.ones(N, M, dtype=torch.bool)
            for j in range(N):
                mask[j, j + N * torch.distributed.get_rank()] = False

            # Use the mask to select the negative samples
            neg_cost = - cost[i][mask].view(N, M-1)
            with torch.no_grad():
                p_cr = torch.softmax(neg_cost / self.T, dim=1)
            cr = (neg_cost * p_cr).sum(1)
            loss += ((ca + cr) * p[i]).sum() 
        return loss


    def forward(self, x1, x2, m, x_local=None, cost='rbf', use_full_mode=False):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
            x_local: local views of images (if swav multi-crops are applied)
            cost: we use RBF kernel cost or Euclidean cost for CACR loss
            use_full_mode: we use light mode or full mode when multi-crops are applied for CACR (whether local crops are forwarded in the momentum encoder).
        Output:
            loss
        """

        # compute features
        q1 = self.predictor(self.base_encoder(x1))
        q2 = self.predictor(self.base_encoder(x2))
        q_list = [q1, q2]

        if x_local is not None:
            q_local = [self.predictor(self.base_encoder(x)) for x in x_local]
            q_list = q_list + q_local
        else:
            q_local = None

        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m)  # update the momentum encoder

            # compute momentum features as targets
            k1 = self.momentum_encoder(x1)
            k2 = self.momentum_encoder(x2)
            k_list = [k1, k2]
            
            if use_full_mode and x_local is not None:
                k_local = [self.momentum_encoder(x) for x in x_local]
                k_list = k_list + k_local
            else:
                k_local = None

        len_k = len(k_list) if use_full_mode else 2
        loss = 0
        for i in range(len_k):
            q_without_current_index = q_list[:i] + q_list[i+1:]      
            if cost == 'euclidean':
                loss += self.cacr_euclideanloss(q_without_current_index, k_list[i])
            else:
                loss += self.cacr_rbfloss(q_without_current_index, k_list[i])

        return loss / len_k


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
