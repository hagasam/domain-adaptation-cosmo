import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import r2_score
import sys
from tqdm import tqdm

def eval_tgt(encoder, classifier, data_loader, device):
    '''
    This function is used to evaluate the performance of the adapted target encoder.
    It also uses the pre-trained classifier
    '''
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
