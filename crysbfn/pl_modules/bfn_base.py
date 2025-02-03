import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import i0e, i1e
from torch_scatter import scatter_mean, scatter_sum
from tqdm import tqdm
from torch.distributions import Normal
from .egnn.egnn_new import EGNN
import numpy as np
import hydra
from absl import logging
from crysbfn.common.data_utils import (
    EPSILON, _make_global_adjacency_matrix, cart_to_frac_coords, mard, lengths_angles_to_volume,
    frac_to_cart_coords, min_distance_sqr_pbc, remove_mean)
import torch.distributions as D

def corrupt_t_pred(self, mu, t, gamma):
    # if t < self.t_min:
    #   return torch.zeros_like(mu)
    # else:
    # eps_pred = self.model()
    t = torch.clamp(t, min=self.t_min)
    # t = torch.ones((mu.size(0),1)).cuda() * t
    eps_pred = self.model(mu, t)
    x_pred = mu / gamma - torch.sqrt((1 - gamma) / gamma) * eps_pred
    return x_pred


class bfnBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(bfnBase, self).__init__(*args, **kwargs)

    # def zero_center_of_mass(self, x_pos, segment_ids):
    #     size = x_pos.size()
    #     assert len(size) == 2  # TODO check this
    #     seg_means = scatter_mean(x_pos, segment_ids, dim=0)
    #     mean_for_each_segment = seg_means.index_select(0, segment_ids)
    #     x = x_pos - mean_for_each_segment
    #     return x
    
    def zero_center_of_mass(self, x_pos, segment_ids):
        size = x_pos.size()
        assert len(size) == 2  # TODO check this
        seg_means = scatter_mean(x_pos, segment_ids, dim=0)
        mean_for_each_segment = seg_means.index_select(0, segment_ids)
        x = x_pos - mean_for_each_segment
        return x

    def get_k_params(self, bins):
        """
        function to get the k parameters for the discretised variable
        """
        # k = torch.ones_like(mu)
        # ones_ = torch.ones((mu.size()[1:])).cuda()
        # ones_ = ones_.unsqueeze(0)
        list_c = []
        list_l = []
        list_r = []
        for k in range(1, int(bins + 1)):
            # k = torch.cat([k,torch.ones_like(mu)*(i+1)],dim=1
            k_c = (2 * k - 1) / bins - 1
            k_l = k_c - 1 / bins
            k_r = k_c + 1 / bins
            list_c.append(k_c)
            list_l.append(k_l)
            list_r.append(k_r)
        # k_c = torch.cat(list_c,dim=0)
        # k_l = torch.cat(list_l,dim=0)
        # k_r = torch.cat(list_r,dim=0)

        return list_c, list_l, list_r

    def discretised_cdf(self, mu, sigma, x):
        """
        cdf function for the discretised variable
        """
        # in this case we use the discretised cdf for the discretised output function
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)  # B,1,D

        f_ = 0.5 * (1 + torch.erf((x - mu) / ((sigma) * np.sqrt(2))))
        flag_upper = torch.ge(x, 1)
        flag_lower = torch.le(x, -1)
        f_ = torch.where(flag_upper, torch.ones_like(f_), f_)
        f_ = torch.where(flag_lower, torch.zeros_like(f_), f_)
        return f_

    def continuous_var_bayesian_flow(self, t, sigma1, x, ret_eps=False, n_samples=1):
        """
        x: [N, D]
        """
        if n_samples == 1:
            gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
            eps = torch.randn_like(x)  # [B, D]
            mu = gamma * x + eps * torch.sqrt(gamma * (1 - gamma))
        else:
            t = t.unsqueeze(-1).repeat(1,1,n_samples)
            x = x.unsqueeze(-1).repeat(1,1,n_samples)
            gamma = 1 - torch.pow(sigma1, 2 * t)
            eps = torch.randn_like(x)
            mu = gamma * x + eps * torch.sqrt(gamma * (1 - gamma))
            mu = mu.mean(-1)
        if not ret_eps:
            return mu, gamma
        else:
            return mu, gamma, eps

    def discrete_var_bayesian_flow(self, t, beta1, x, K, ret_eps=False):
        """
        x: [N, K]
        """
        beta = beta1 * (t**2)  # (B,)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        if not ret_eps:
            return theta
        else:
            return theta, eps

    def ctime4continuous_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    # def ctime4continuous_loss(self, t, sigma1, x_pred, x, pbc_dist=None):
    #     if pbc_dist == None:
    #         loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
    #     else:
    #         loss = pbc_dist.view(x.shape[0], -1).abs().pow(2).sum(dim=1)
    #     return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))



    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x, segment_ids=None, mult_constant=True, wn=False):
        # TODO not debuged yet
        if wn:
            steps = torch.arange(1,N+1).to(self.device)
            all_weights = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * steps / N))
            weight_norm = all_weights.mean().detach().clone()
        else:
            weight_norm = 1.
        if segment_ids is not None:
            weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
            loss = scatter_mean(weight.view(-1)*((x_pred - x)**2).mean(-1),segment_ids,dim=0)
        else:
            if mult_constant:
                loss =  N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N)) * (x_pred - x).view(x.shape[0], -1).abs().pow(2)
            else:
                loss =  (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N)) * (x_pred - x).view(x.shape[0], -1).abs().pow(2)
        # print(loss.shape)
        return loss.mean() / weight_norm
    
    def dtime4continuous_loss_cir(self, i, N, sigma1, x_pred, x, segment_ids=None, mult_constant=True, wn=False):
        freqs = torch.arange(-10,11).to(self.device).unsqueeze(0).unsqueeze(0)*np.pi*2
        tar_coord = x.unsqueeze(-1) + freqs
        coord_diff = (tar_coord - x_pred.unsqueeze(-1)).square().min(dim=-1).values
        if wn:
            steps = torch.arange(1,N+1).to(self.device)
            all_weights = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * steps / N))
            weight_norm = all_weights.mean().detach().clone()
        else:
            weight_norm = 1.
        if segment_ids is not None:
            weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
            loss = scatter_mean(weight.view(-1)*((x_pred - x)**2).mean(-1),segment_ids,dim=0)
        else:
            if mult_constant:
                loss =  N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N)) * coord_diff.view(x.shape[0], -1).abs()
            else:
                loss =  (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N)) * coord_diff.view(x.shape[0], -1).abs()
        # print(loss.shape)
        return loss.mean() / weight_norm

    def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K):
        e_x = one_hot_x  # [N, K]
        e_hat = p_0  # (N, K)
        L_infinity = K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1)
        return L_infinity.mean()

    def ctime4discreteised_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    # def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None, mult_constant=True):
    #     alpha = beta1 * (2 * i - 1) / (N**2)  # [N]
    #     e_x = one_hot_x  # [N, T, K]
    #     e_hat = p_0  # (N, T, K)
    #     if mult_constant:
    #         L_n = N * (K * alpha * (((e_x - e_hat) ** 2).sum(dim=-1)))
    #     else:
    #         L_n = (K * alpha * (((e_x - e_hat) ** 2).sum(dim=-1)))
    #     # L_n = N * (K * alpha * (((e_x - e_hat) ** 2).sum(dim=2).sum(dim=1))).mean()
    #     return L_n

    # def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None,mult_constant=True):
    #     if not mult_constant:
    #         N = 1
    #     # i in {1,n}
    #     # Algorithm 7 in BFN
    #     D = one_hot_x.size()[0]
    #     e_x = one_hot_x  # [D, K]
    #     e_hat = p_0  # (D, K)
    #     assert e_x.size() == e_hat.size()
    #     alpha = beta1 * (2 * i - 1) / N**2  # [D]
    #     mean_ = alpha * (K * e_x - 1)  # [D, K]
    #     std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
    #     eps = torch.randn_like(mean_)  # [D,K,]
    #     y_ = mean_ + std_ * eps  # [D, K]
    #     matrix_ek = torch.eye(K, K).to(e_x.device).unsqueeze(0).repeat(D,1,1)  # [D, K, K]
    #     mean_matrix = alpha.unsqueeze(-1) * (K * matrix_ek - 1)  # [K, K]
    #     std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
    #     LOG2PI = torch.log(torch.tensor(2 * np.pi))
    #     _log_gaussians = (  # [D, K]
    #         (-0.5 * LOG2PI - torch.log(std_matrix))
    #         - (y_.unsqueeze(1) - mean_matrix) ** 2 / (2 * std_matrix**2)
    #     ).sum(-1)
    #     _inner_log_likelihood = torch.log(torch.sum(e_hat * torch.exp(_log_gaussians), dim=-1))  # (D,)
    #     if segment_ids is not None:
    #         L_N = -scatter_mean(_inner_log_likelihood, segment_ids, dim=0)
    #     else:
    #         L_N = -_inner_log_likelihood.sum(dim=-1)  # [D]
    #     return N * L_N

    def dtime4discrete_loss_prob(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, segment_ids=None, time_scheduler ="quad", beta_init=None
    ):
        # this is based on the official implementation of BFN.
        # import pdb
        # pdb.set_trace()
        target_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D,  K)

        if time_scheduler == "quad":
            alpha = beta1 * (2 * i - 1) / (N**2)  # [N]
        elif time_scheduler == "linear":
            alpha = beta1 / N
        elif time_scheduler == "hybrid":
            assert beta_init is not None
            alpha = (beta1 - beta_init) * (2 * i - 1) / (N**2) + beta_init / N     
        else:
            raise NotImplementedError   
        
        alpha = alpha.view(-1, 1) # [D, 1]

        classes = torch.arange(K, device=target_x.device).long().unsqueeze(0)  # [ 1, K]
        e_x = F.one_hot(classes.long(), K) #[1,K, K]
        # print(e_x.shape)
        receiver_components = D.Independent(
            D.Normal(
                alpha.unsqueeze(-1) * ((K * e_x) - 1), # [D K, K]
                (K * alpha.unsqueeze(-1)) ** 0.5, # [D, 1, 1]
            ),
            1,
        )  # [D,T, K, K]
        receiver_mix_distribution = D.Categorical(probs=e_hat)  # [D, K]
        receiver_dist = D.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )  # [D, K]
        # pdb.set_trace()
        # print(receiver_dist.event_shape)

        sender_dist = D.Independent( D.Normal(
            alpha* ((K * target_x) - 1), ((K * alpha) ** 0.5)
        ),1)  # [D, K]

        y = sender_dist.sample(torch.Size([n_samples])) 

        # print(sender_dist.log_prob(y).size())
        # import pdb
        # pdb.set_trace()

        loss = N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).mean(
            -1, keepdims=True
        )

        # loss = (
        #         (sender_dist.log_prob(y) - receiver_dist.log_prob(y))
        #         .mean(0)
        #         .flatten(start_dim=1)
        #         .mean(1, keepdims=True)
        #     )
        # #
        return loss.mean()




    def dtime4discrete_loss(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, segment_ids=None
    ):
        if K == 1:
            return torch.zeros_like(one_hot_x)
            n_samples = 5
        # this is based on the official implementation of BFN.
        target_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D,  K)
        alpha = beta1 * (2 * i - 1) / (N**2)  # [D]
        alpha = alpha.view(-1, 1) # [D, 1]
        classes = torch.arange(K, device=target_x.device).long().unsqueeze(0)  # [ 1, K]
        e_x = F.one_hot(classes.long(), K) #[1,K, K]
        # print(e_x.shape)
        receiver_components = D.Independent(
            D.Normal(
                alpha.unsqueeze(-1) * ((K * e_x) - 1), # [D K, K]
                (K * alpha.unsqueeze(-1)) ** 0.5, # [D, 1, 1]
            ),
            1,
        )  # [D,T, K, K]
        receiver_mix_distribution = D.Categorical(probs=e_hat)  # [D, K]
        receiver_dist = D.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )  # [D, K]
        sender_dist = D.Independent( D.Normal(
            alpha* ((K * target_x) - 1), ((K * alpha) ** 0.5)
        ),1)  # [D, K]
        y = sender_dist.sample(torch.Size([n_samples])) 
        # if segment_ids != None:
        #     loss = scatter_mean(N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0), segment_ids, dim=0)
        # else:
        loss = N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).mean(
            -1, keepdims=True
        )
        
        return loss.mean()

    def dtime4circular_loss(self, i, N, alpha_i, x_pred, x, segment_ids=None, mult_constant=True, weight_norm=1, wn=False, mse_loss=True):
        if not mult_constant:
            N = 1
        freqs = torch.arange(-10,11).to(self.device).unsqueeze(0).unsqueeze(0)*np.pi*2
        tar_coord = x.unsqueeze(-1) + freqs
        coord_diff = (tar_coord - x_pred.unsqueeze(-1)).square().min(dim=-1).values
        weight = (i1e(alpha_i) / i0e(alpha_i)) * alpha_i   
        if mse_loss:
            loss = N * weight * coord_diff
        else:
            loss = N * weight * (1 - torch.cos(x_pred - x))
        if not wn:
            weight_norm = 1.
        return loss.mean() / weight_norm


    def ctime4circular_loss(self, t, beta1, x_pred, x, segment_ids,):
        alpha_t = beta1.pow(t) * (1 + t * torch.log(beta1))
        weight = i1e(alpha_t) * alpha_t / i0e(alpha_t)
        loss = 1 - torch.cos(x_pred - x)
        if segment_ids != None:
            return scatter_mean((weight * loss).sum(-1), segment_ids, dim=0)
        else:
            return (weight * loss).sum(-1)

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

