import torch.nn as nn
import torch.optim as optim
import torch
import params
import numpy as np
import sys
from sklearn.metrics import r2_score
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR, StepLR, ReduceLROnPlateau


def loss_camels(local_labels, y_NN, e_NN):
    loss1 = torch.mean((y_NN - local_labels)**2, axis=0)
    loss2 = torch.mean(((y_NN - local_labels)**2 - e_NN**2)**2, axis=0)
    loss  = torch.mean(torch.log(loss1) + torch.log(loss2))
    return loss

def train_src(encoder, classifier, data_loader, device, num_epochs = 201):
    """Train source network on source domain."""
    ####################
    # 1. setup network #
    ####################
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr = 0.0015, amsgrad = True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    

    ####################
    # 2. train network #
    ####################

    for epoch in range(num_epochs):
        train_loss = 0.0
        with tqdm(total=len(data_loader), ncols = 100) as progress_bar:
            for step, (images, labels) in enumerate(data_loader):
                # make images and labels variable
                labels = labels.to(device)
                images = images.to(device)

                # zero gradients for optimizer
                optimizer.zero_grad()

                # compute loss for critic
                preds = classifier(encoder(images))
                loss = loss_camels(labels, preds[:,:2], preds[:,2:]) # loss from camels

                # optimize source classifier
                loss.backward()
                optimizer.step()
                
                progress_bar.set_description(str(epoch)) 
                progress_bar.set_postfix(loss = loss.item())
                progress_bar.update(1) 
                train_loss += loss.item()
    # # save final model
        scheduler.step(train_loss / float(step + 1))
        if epoch != 0 and epoch % 100 == 0:
            # TNG -> SIMBA
            # torch.save(encoder.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_encfeats_tng_%d.pt' % epoch)
            # torch.save(classifier.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_regfeats_tng_%d.pt' % epoch)
            # SIMBA -> TNG
            torch.save(encoder.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_encfeats_simba_%d.pt' % epoch)
            torch.save(classifier.state_dict(),'/home/handrianomena/research/projects/camels/Regression/domain-adaptation/models/adda_regfeats_simba_%d.pt' % epoch)
    return encoder, classifier


def eval_src(encoder, classifier, data_loader, device):
    """
    This function is used to assess the performance of the source network (i.e. encoder + classifier)
    """
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()
    # set loss function
    # evaluate network
    pred_mean = []
    pred_err = []
    y_test = []
    with torch.no_grad():
        for (images, labels) in tqdm(data_loader, ncols = 100):
            images  = images.to(device)
            prediction_ = classifier(encoder(images))
            y_NN  = prediction_[:,:2]       #prediction for mean
            e_NN  = prediction_[:,2:]       #prediction for error
            pred_mean.append(y_NN.cpu().numpy())
            pred_err.append(e_NN.cpu().numpy())
            y_test.append(labels.numpy())
    pred_mean = np.array(pred_mean) 
    pred_err = np.array(pred_err) 
    y_test = np.array(y_test)
    pred_err = pred_err.reshape(-1, 2)
    pred_mean, y_test = pred_mean.reshape(-1, 2), y_test.reshape(-1, 2)
    minimum = np.array([0.1, 0.6])
    maximum = np.array([0.5, 1.0])
    true_test = y_test * (maximum - minimum) + minimum
    predictions = pred_mean * (maximum - minimum) + minimum
    print('r2 omega:%3.3f | sig8:%3.3f' % (r2_score(true_test[:,0], predictions[:,0]), r2_score(true_test[:,1], predictions[:,1])))
    
 