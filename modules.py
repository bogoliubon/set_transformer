import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class miniset_MAB(nn.Module):
    """ input batch_size * set_size * dim, output batch_size * miniset * dim
    """
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, miniset:int, minisettype='miniset_A1', ln=False, flash=False):
        super(miniset_MAB, self).__init__()
        self.minisettype = minisettype
        self.mab1 = MAB(dim_Q, dim_K, dim_V, num_heads, ln=ln)
        if self.minisettype == 'miniset_A4':
            self.mab2 = MAB(dim_V, dim_Q, dim_V, num_heads, ln=ln)
        else: 
            self.mab2 = MAB(dim_V, dim_V, dim_V, num_heads, ln=ln)
        self.miniset = miniset

    def forward(self, X):
        # import pdb; pdb.set_trace()
        assert X.shape[1] % self.miniset == 0, "set size must be divided by mini set size!"
        n_mini = X.shape[1] // self.miniset
        inter_output_list = []

        if self.minisettype=='miniset_A1':
            for i in range(n_mini):
                curr = X[:,i*self.miniset: (i+1)*self.miniset, :]
                next = X[:,(i+1)%n_mini*self.miniset: (i+2)%n_mini*self.miniset, :]
                inter_output_list.append(self.mab1(curr, next))
            del curr

            mid_output = []
            for i in range(n_mini):
                curr = inter_output_list[i]
                for j in range(1,n_mini):
                    next = inter_output_list[(j+i)%n_mini]
                    curr = self.mab2(curr, next)
                mid_output.append(curr)
            del inter_output_list
            random.shuffle(mid_output)
            output = mid_output[0]
            for i,mat in enumerate(mid_output[1:]):
                output = self.mab2(output, mat)
            del mid_output

        elif self.minisettype=='miniset_A2':
            for i in range(n_mini):
                curr = X[:,i*self.miniset: (i+1)*self.miniset, :]
                inter_output_list.append(self.mab1(curr, curr))
            del curr

            mid_output = []
            for i in range(n_mini):
                curr = inter_output_list[i]
                for j in range(1,n_mini):
                    next = inter_output_list[(j+i)%n_mini]
                    curr = self.mab2(curr, next)
                mid_output.append(curr)
            del inter_output_list
            random.shuffle(mid_output)
            output = mid_output[0]
            for i,mat in enumerate(mid_output[1:]):
                output = self.mab2(output, mat)
            del mid_output

        elif self.minisettype=='miniset_A3':
            # import pdb; pdb.set_trace()
            for i in range(X.shape[1] // self.miniset-1):
                curr = X[:,i*self.miniset: (i+1)*self.miniset, :]
                next = X[:,(i+1)*self.miniset: (i+2)*self.miniset, :]
                inter_output_list.append(self.mab1(curr, next))
            inter_output_list.append(self.mab1(next, X[:,:self.miniset, :]))
            random.shuffle(inter_output_list)
            output = inter_output_list[0]
            for i,mat in enumerate(inter_output_list[1:]):
                output = self.mab2(output, mat)

        elif self.minisettype=='miniset_A4':
            mini_list = [X[:, i*self.miniset: (i+1)*self.miniset, :] for i in range(n_mini)]
            output = self.mab1(mini_list[0], mini_list[1])
            for next in mini_list[1:]:
                output = self.mab2(output, next)

        else:
            raise ValueError("model not implemented")
            
        return output




class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)
