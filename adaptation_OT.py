from numpy.random import seed
seed(1)
import torch
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import torchvision
from torch.utils import data
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import sys
import time
from torch.optim.lr_scheduler import ExponentialLR, StepLR, ReduceLROnPlateau
import random
random.seed(999)
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import torchvision.models
from models import Discriminator, CNNRegressor, Encoder
from utils import get_data_loader, init_model, init_random_seed
import ot


# init_random_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# load dataset
# TNG -> SIMBA
# path_feats = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/feats_tng.npz'
# tgt_data_loader = get_data_loader('SIMBA')
# params_train = {'batch_size': 50, 'shuffle': True}
# data_feats_src = np.load(path_feats)
# feat_src, label_src = data_feats_src['feats'], data_feats_src['labels']
# feat_set = torch.utils.data.TensorDataset(torch.tensor(feat_src.astype(np.float32)), torch.tensor(label_src.astype(np.float32)))
# feat_loader = torch.utils.data.DataLoader(feat_set, **params_train)

# SIMBA -> TNG
path_feats = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/feats_simba.npz'
tgt_data_loader = get_data_loader('IllustrisTNG')
params_train = {'batch_size': 50, 'shuffle': True}
data_feats_src = np.load(path_feats)
feat_src, label_src = data_feats_src['feats'], data_feats_src['labels']
feat_set = torch.utils.data.TensorDataset(torch.tensor(feat_src.astype(np.float32)), torch.tensor(label_src.astype(np.float32)))
feat_loader = torch.utils.data.DataLoader(feat_set, **params_train)

epoch_tgt = 200#100
path_tgenc = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_encfeats_simba_%d.pt' % epoch_tgt # this is for strating training
tgt_encoder = init_model(net=Encoder(), restore=path_tgenc)

optimizer = optim.Adam(tgt_encoder.parameters(), lr=1e-5, amsgrad=True) 

BATCH_SIZE = 50
ab = torch.ones(BATCH_SIZE) / BATCH_SIZE
ab = ab.to(device)

loss_epoch = []
for epoch in tqdm(range(201), ncols = 75):
    losses = 0.0
    i = 1
    for i, data in enumerate(zip(feat_loader, tgt_data_loader)):
        src_feat, label_src = data[0] # IllustrisTNG/SIMBA
        tgt_im, label_tgt = data[1] # SIMBA/IllustrisTNG
        pred_tgt_src = tgt_encoder(tgt_im.to(device)) 
        M = ot.dist(pred_tgt_src, src_feat.to(device))
        loss = ot.emd2(ab, ab, M)
        loss.backward()
        optimizer.step()
        losses += loss.item()
    loss_epoch.append(losses / float(i))
    if epoch != 0 and epoch % 25 == 0:
        # torch.save(tgt_encoder.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_tgtencfeats_ot_tng3_%d.pt' % (epoch + 100))
        torch.save(tgt_encoder.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_tgtencfeats_ot_simba3_%d.pt' % (epoch))