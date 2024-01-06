import argparse
import os

import numpy as np
import torch
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

from data_modelnet40 import ModelFetcher
from modules import ISAB, PMA, SAB
from models import *

import wandb
from tqdm import tqdm

class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input=3,
        num_outputs=1,
        dim_output=40,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            nn.Dropout(),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X)).squeeze()


parser = argparse.ArgumentParser()

parser.add_argument("--num_pts", type=int, default=10000)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--batch_size", type=int, default=64)

parser.add_argument("--model", type=str, default='ST', help='options: ST, miniST')

parser.add_argument("--miniset", type=int, default=2000)
parser.add_argument("--miniset_type", type=str)

parser.add_argument("--dim", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=4)
parser.add_argument("--n_anc", type=int, default=16)
parser.add_argument("--train_epochs", type=int, default=100)

parser.add_argument("--debug", action='store_true', default=False)
args = parser.parse_args()
# log_dir = "result/" + args.exp_name
# model_path = log_dir + "/model"

# import pdb; pdb.set_trace()

generator = ModelFetcher(
    "ModelNet40_cloud.h5",
    args.batch_size,
    down_sample=int(10000 / args.num_pts),
    do_standardize=True,
    do_augmentation=(args.num_pts == 5000),
)

if args.model == 'ST':  
    model = SetTransformer(dim_hidden=args.dim, num_heads=args.n_heads, num_inds=args.n_anc)
    args.exp_name = f"ST_N{args.num_pts}_d{args.dim}h{args.n_heads}i{args.n_anc}_lr{args.learning_rate}bs{args.batch_size}"

elif args.model == 'miniST':
    model = SetTransformer_miniSAB(dim_input=3, set_size=args.num_pts , dim_output=40, dim_hidden=args.dim,
                                   num_heads=args.n_heads, p_outputs=1, miniset=args.miniset, minisettype=args.miniset_type, model_loaded=None, ln=True, flash=False)
    args.exp_name = f"miniST_N{args.num_pts}_d{args.dim}h{args.n_heads}{args.miniset_type}mini{args.miniset}_lr{args.learning_rate}bs{args. batch_size}"
                               
else:
    raise ValueError('model not implemented.')

# import pdb; pdb.set_trace()

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

if not args.debug:
    wandb.init(project='pointcloud classification miniset', name=args.exp_name)
    wandb.log({'# of parameters': total_params})

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
criterion = nn.CrossEntropyLoss()
model = nn.DataParallel(model)
model = model.cuda()

for epoch in range(args.train_epochs):
    model.train()
    losses, total, correct = [], 0, 0
    for imgs, _, lbls in tqdm(generator.train_data(), desc=f"epoch{epoch}: training..."):
        imgs = torch.Tensor(imgs).cuda()
        lbls = torch.Tensor(lbls).long().cuda()
        preds = model(imgs)
        loss = criterion(preds, lbls)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        total += lbls.shape[0]
        correct += (preds.argmax(dim=1) == lbls).sum().item()

    avg_loss, avg_acc = np.mean(losses), correct / total
    print(f"Epoch {epoch}: train loss {avg_loss:.3f} train acc {avg_acc:.3f}")


    model.eval()
    losses, total, correct = [], 0, 0
    for imgs, _, lbls in tqdm(generator.test_data(), desc= f"epoch{epoch}: evaluating..."):
        imgs = torch.Tensor(imgs).cuda()
        lbls = torch.Tensor(lbls).long().cuda()
        # import pdb; pdb.set_trace()
        preds = model(imgs)
        loss = criterion(preds, lbls)

        losses.append(loss.item())
        total += lbls.shape[0]
        correct += (preds.argmax(dim=1) == lbls).sum().item()
    avg_loss, avg_acc = np.mean(losses), correct / total
    wandb.log({"epoch": epoch, "train_loss": avg_loss, "train_acc": avg_acc, "test_loss": avg_loss, "test_acc": avg_acc})
    print(f"Epoch {epoch}: test loss {avg_loss:.3f} test acc {avg_acc:.3f}")
