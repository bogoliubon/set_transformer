from modules import *
import random

class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
                nn.Linear(dim_input, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, dim_hidden),
                nn.ReLU(),
                nn.Linear(dim_hidden, num_outputs*dim_output))

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X

class SetTransformer(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output,
            num_inds=32, dim_hidden=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
                ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln))
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))

class SetTransformer_miniSAB_new(nn.Module):
    def __init__(self, dim_input, set_size, dim_output, \
                dim_hidden, num_heads, p_outputs, miniset:int, \
                minisettype, num_outputs=1, model_loaded=None, ln=True, flash=False):
        super(SetTransformer_miniSAB_new,self).__init__()
        self.set_size = set_size
        self.miniset = miniset
        
        self.enc = miniset_MAB(dim_input, dim_input, dim_hidden, \
                                num_heads, miniset, minisettype, ln=ln, flash=False)

        if model_loaded:
            self.enc = model_loaded.enc
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, num_outputs, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
                nn.Linear(dim_hidden, dim_output))

    def forward(self, X):
        return self.dec(self.enc(X))

class SetTransformer_miniSAB(nn.Module):
    def __init__(self, dim_input, set_size, dim_output, \
                dim_hidden, num_heads, p_outputs, miniset:int, \
                minisettype, model_loaded=None, ln=True, flash=False):
        super(SetTransformer_miniSAB,self).__init__()
        self.set_size = set_size
        self.miniset = miniset
        self.sig = nn.Sigmoid()
        
        self.enc = miniset_MAB(dim_input, dim_input, dim_hidden, \
                                num_heads, miniset, minisettype, ln=ln, flash=False)

        if model_loaded:
            self.enc = model_loaded.enc
        
        self.dec = nn.Sequential(
                PMA(dim_hidden, num_heads, p_outputs, ln=ln),
                nn.Linear(dim_hidden, dim_output),
                nn.Sigmoid())

    def forward(self, X):
        # import pdb; pdb.set_trace()
        X = self.sig(self.enc(X))
        return self.dec(X)[:,0]
