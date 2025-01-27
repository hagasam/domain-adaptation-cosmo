import torch
from torchvision import datasets, transforms
# import params
import numpy as np
import random
random.seed(42)
from scipy.ndimage.filters import gaussian_filter


def read_data(fileparams):
    maps_ = np.load('/home/handrianomena/research/data/Maps_HI_IllustrisTNG_LH_z=0.00.npy')
    maps_ = np.log10(maps_)
    min_maps, max_maps = maps_.min(), maps_.max()
    maps__ = (maps_ - min_maps) / (max_maps - min_maps)
    maps = maps__[:,None,:,:]    
    params = np.loadtxt(fileparams)
    labels = np.zeros((len(maps), 2))
    for i in range(len(maps)):
        labels[i] = np.array(params[i // 15][[0,1]])
    # normalize params
    minimum = np.array([0.1, 0.6])
    maximum = np.array([0.5, 1.0])
    labels  = (labels - minimum)/(maximum - minimum)
    ind_sample = random.sample(range(0, labels.shape[0]), labels.shape[0])
    return maps[ind_sample], labels[ind_sample] # shuffling the data

def augment_data(train_x, train_y):
    final_train_data = []
    final_target_train = []
    for i in range(train_x.shape[0]):
        final_train_data.append(train_x[i])
        final_train_data.append(np.fliplr(gaussian_filter(train_x[i], sigma=1.3)))
        for j in range(2):
            final_target_train.append(train_y[i])
    final_train = np.array(final_train_data)
    final_target_train = np.array(final_target_train)
    return final_train, final_target_train

        
def get_illustris(train = True):
    fileparams = '/home/handrianomena/research/data/params_IllustrisTNG.txt'
    maps, labels = read_data(fileparams)
    ind = 3000
    params_train = {'batch_size': 50, 'shuffle': True}
    params_test = {'batch_size': 50, 'shuffle': False}
    # augment the training data
    if train:
        x, y = maps[:-ind], labels[:-ind]
        train_x, train_y = augment_data(x, y)
        train_set = torch.utils.data.TensorDataset(torch.tensor(train_x.astype(np.float32)), torch.tensor(train_y.astype(np.float32)))
        train_loader = torch.utils.data.DataLoader(train_set, **params_train)
        return train_loader
    else:
        test_set = torch.utils.data.TensorDataset(torch.tensor(maps[-ind:-ind//2].astype(np.float32)), torch.tensor(labels[-ind:-ind//2].astype(np.float32)))
        test_loader = torch.utils.data.DataLoader(test_set, **params_test)    
        return test_loader

def get_illustris_inv(train = True):
    maps_list = []
    fileparams = '/home/handrianomena/research/data/params_IllustrisTNG.txt'
    maps_ = np.load('/home/handrianomena/research/data/Maps_HI_IllustrisTNG_LH_z=0.00.npy')
    maps_ = np.log10(maps_) 
    maps__ = (maps_ - maps_.min())/(maps_.max() - maps_.min())
    maps = maps__[:,None,:,:]
    params = np.loadtxt(fileparams)
    labels = np.zeros((len(maps), 2))
    for i in range(len(maps)):
        labels[i] = np.array(params[i // 15][[0,1]])
    # normalize params
    minimum = np.array([0.1, 0.6])
    maximum = np.array([0.5, 1.0])
    labels  = (labels - minimum)/(maximum - minimum)
    ind_sample = random.sample(range(0, labels.shape[0]), labels.shape[0])
    if train:
        maps_, labels_ = maps[ind_sample][:-3000], labels[ind_sample][:-3000] # this works
        shuffle = True
    else:
        maps_, labels_ = maps[ind_sample][-3000:-1500], labels[ind_sample][-3000:-1500] # shuffling the data
        shuffle = False    
    params_set = {'batch_size': 50, 'shuffle': shuffle}
    # params_set = {'batch_size': 15, 'shuffle': shuffle}
    dataset = torch.utils.data.TensorDataset(torch.tensor(maps_.astype(np.float32)), torch.tensor(labels_.astype(np.float32)))
    data_loader = torch.utils.data.DataLoader(dataset, **params_set)
    return data_loader





