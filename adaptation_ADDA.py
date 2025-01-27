from numpy.random import seed
seed(1)
import torch
torch.backends.cudnn.deterministic = True
torch.manual_seed(999)
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
# import params
from core import eval_src, eval_tgt, train_src, train_tgt
from models import Discriminator, CNNRegressor, Encoder 
from utils import get_data_loader, init_model, init_random_seed
import sys
from sklearn.metrics import r2_score
import time

init_random_seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# TNG -> SIMBA
# src_data_loader = get_data_loader('IllustrisTNG')
# src_data_loader_eval = get_data_loader('IllustrisTNG', train=False)
# tgt_data_loader = get_data_loader('SIMBA')
# tgt_data_loader_eval = get_data_loader('SIMBA', train=False)

# SIMBA -> TNG
src_data_loader = get_data_loader('SIMBA')
src_data_loader_eval = get_data_loader('SIMBA', train=False)
tgt_data_loader = get_data_loader('IllustrisTNG')
tgt_data_loader_eval = get_data_loader('IllustrisTNG', train=False)
print('data loaded ...')

epoch = 200
# path_enc = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_enc_tng_%d.pt' % epoch
# path_reg = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_reg_tng_%d.pt' % epoch

# TNG -> SIMBA
# path_enc = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_encfeats_tng_%d.pt' % epoch
# path_reg = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_regfeats_tng_%d.pt' % epoch

# SIMBA -> TNG
path_enc = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_encfeats_simba_%d.pt' % epoch
path_reg = '/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_regfeats_simba_%d.pt' % epoch


src_encoder = init_model(Encoder(), restore = path_enc)
src_classifier = init_model(CNNRegressor(), restore = path_reg)
tgt_encoder = init_model(net=Encoder(), restore=None)
critic = init_model(Discriminator(input_dims=512,
                                      hidden_dims=512,
                                      output_dims=2),restore=None)
print('models loaded ...')
print('training target begins ...')
start = time.time()
tgt_encoder.load_state_dict(src_encoder.state_dict()) # transferring weights to target network
tgt_encoder = train_tgt(src_encoder, tgt_encoder, critic, src_data_loader, tgt_data_loader, device) # uncomment if training is needed
print('training takes about ', (time.time() - start) / (60 * 60), ' hours')
