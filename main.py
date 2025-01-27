from numpy.random import seed
seed(1)
import torch
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, CNNRegressor, Encoder
from utils import get_data_loader, init_model, init_random_seed
import sys
from sklearn.metrics import r2_score
import time

init_random_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# load dataset
# TNG -> SIMBA
# src_data_loader = get_data_loader('IllustrisTNG') 
# src_data_loader_eval = get_data_loader('IllustrisTNG', train=False)
# SIMBA -> TNG
src_data_loader = get_data_loader('SIMBA')
src_data_loader_eval = get_data_loader('SIMBA', train=False)

print('dataset loaded ...')
#path_enc = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_enc_tng_100.pt'
#path_reg = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_reg_tng_100.pt'
# src_encoder = init_model(CNNEncoder(), restore = None)
src_encoder = init_model(Encoder(), restore = None)
src_classifier = init_model(CNNRegressor(), restore = None)
print('models loaded ...')
start = time.time()
src_encoder, src_classifier = train_src(src_encoder, src_classifier, src_data_loader, device)
print('training takes about ', (time.time() - start) / (60 * 60), ' hours')
