'''
This is based on this repo: https://github.com/corenel/pytorch-adda/tree/master
'''

import os
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR, StepLR, ReduceLROnPlateau
import sys


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, device, num_epochs = 1001):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=1e-5,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=1e-4,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(num_epochs):
        data_zip = zip(src_data_loader, tgt_data_loader)
        with tqdm(total=len(src_data_loader), ncols = 100) as progress_bar:
            for step, ((images_src, _), (images_tgt, _)) in enumerate(data_zip):
                ###########################
                # 2.1 train discriminator #
                ###########################

                # make images variable
                images_src = images_src.to(device)
                images_tgt = images_tgt.to(device)

                # zero gradients for optimizer
                optimizer_critic.zero_grad()
                optimizer_tgt.zero_grad()

                # extract and concat features
                feat_src = src_encoder(images_src)
                feat_tgt = tgt_encoder(images_tgt)
                feat_concat = torch.cat((feat_src, feat_tgt), 0)
                pred_concat = critic(feat_concat.detach())

                # prepare real and fake label
                label_src = torch.ones(feat_src.size(0)).long()
                label_tgt = torch.zeros(feat_tgt.size(0)).long()
                label_concat = torch.cat((label_src, label_tgt), 0)

                # compute loss for critic
                loss_critic = criterion(pred_concat, label_concat.to(device))
                loss_critic.backward()

                # optimize critic
                optimizer_critic.step()
                

                ############################
                # 2.2 train target encoder #
                ############################

                # extract and target features
                feat_tgt = tgt_encoder(images_tgt)

                # predict on discriminator
                #pred_tgt = critic(feat_tgt.reshape(-1, 2, 16, 16))
                pred_tgt = critic(feat_tgt)

                # prepare fake labels
                label_tgt = torch.ones(feat_tgt.size(0)).long()
                # compute loss for target encoder
                loss_tgt = criterion(pred_tgt, label_tgt.to(device))
                loss_tgt.backward()

                # optimize target encoder
                optimizer_tgt.step()
                
                
                progress_bar.set_description(str(epoch)) 
                progress_bar.set_postfix(loss_t = loss_tgt.item(), loss_c = loss_critic.item()) 
                progress_bar.update(1)

        #############################
        # 2.4 save model parameters #
        #############################
        if epoch != 0 and epoch % 200 == 0:
            # torch.save(tgt_encoder.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_tgtenc_tng3_%d.pt' % epoch)
            # torch.save(critic.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_cri_tng3_%d.pt' % epoch)
            # TNG -> SIMBA
            # torch.save(tgt_encoder.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_tgtencfeats_moredata_tng3_%d.pt' % epoch)
            # torch.save(critic.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_crifeats_moredata_tng3_%d.pt' % epoch)
            # SIMBA -> TNG
            torch.save(tgt_encoder.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_tgtencfeats_moredata_simba3_%d.pt' % epoch)
            torch.save(critic.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_crifeats_moredata_simba3_%d.pt' % epoch)
    return tgt_encoder
