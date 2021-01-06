#!/usr/bin/env python3
#coding=utf8
import torch
import torch.nn.functional as F
import numpy as np

class LogManager:
    def __init__(self):
        self.log_book=dict()

    def alloc_stat_type(self, stat_type):
        self.log_book[stat_type] = []

    def alloc_stat_type_list(self, stat_type_list):
        for stat_type in stat_type_list:
            self.alloc_stat_type(stat_type)

    def init_stat(self):
        for stat_type in self.log_book.keys():
            self.log_book[stat_type] = []

    def add_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat)
        
    def add_torch_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat.detach().cpu().item())
        
    def get_stat(self, stat_type):
        result_stat = 0
        stat_list = self.log_book[stat_type]
        if len(stat_list) != 0:
            result_stat = np.mean(stat_list)
            result_stat = np.round(result_stat, 4)
        return result_stat

    def print_stat(self):
        for stat_type in self.log_book.keys():
           stat = self.get_stat(stat_type)
           if stat != 0:
            print(stat_type,":",stat, end=' / ')
        print(" ")

    


def l1loss(pred, true):
    return F.l1_loss(pred, true)


def l2loss(pred, true):
    """ mean of loss^2 """
    return F.mse_loss(pred, true)

def calc_gaussprob(x, mu, logvar):
    c = torch.log(2.*torch.from_numpy(np.array(3.141592)))
    
    var = logvar.exp()
    x_mu2 = (x - mu).pow(2)  
    x_mu2_over_var = torch.div(x_mu2, var + 1e-6)
    log_prob = -0.5 * (c + logvar + x_mu2_over_var)
    
    return torch.mean(log_prob)

def calc_kl_vae(mu, logvar):    
    dimwise_kld = .5 * (mu.pow(2)+logvar.exp()-1.0-logvar)
    
    return torch.mean(dimwise_kld)

def calc_kl_vae_temp(mu, logvar):    
    dimwise_kld = .5 * (mu.pow(2)+logvar.exp().pow(2)-1.0-2*logvar)
    
    return torch.mean(dimwise_kld)

def nllloss(pred, true, is_batch=False):
    if is_batch:
        """
        pred: (batch, T, softmax)
        true: (batch, T)
        """
        total_mean = 0.0
        for p, t in zip(pred, true):
            total_mean += F.nll_loss(p, t)            
        return total_mean / pred.size()[0]
    else:
        
        return F.nll_loss(pred, true)

def calc_entropy(x):
    # (4, 8, 1, 32)
    x = x.view(-1, x.size()[1])
    x = torch.exp(x) / (torch.sum(torch.exp(x), dim=1).view(-1, 1))
    H = -torch.mean(torch.sum(x*torch.log2(x), dim=1))
    return H

def calc_err(pred, true, is_batch=False):
    if is_batch:
        total_err = 0.0
        for p, t in zip(pred, true):
            ans = torch.argmax(p,dim=1)
            # ans = torch.argmax(p,dim=1)[1]
            err = torch.mean((t!=ans).float())
            total_err += err            
        return total_err / pred.size()[0]
    else:
        ans = torch.argmax(pred,dim=1)
        # ans = torch.argmax(pred,dim=1)[1]
        err = torch.mean((true!=ans).float())
        return err

def calc_entropy_log(x):
    # (4, 4)
    H = -torch.mean(torch.sum(torch.exp(x)*x, dim=1))
    return H

def gradient_penalty(D, pred, true, eps=0.01):
    test_x = (eps*true + (1-eps)*pred).detach()
    test_x.requires_grad=True

    test_loss = D(test_x).mean()
    test_loss.backward()
    test_x.grad.volatile = False

    gp = torch.mean((test_x.grad-1) ** 2)
    return gp