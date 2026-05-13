import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import argparse
import scipy.io as sio

import os
from sklearn.cluster import KMeans
from model import JointNMF
from eval import *
from tools import *

Dataname = 'Diabets'

accs = []
nmis = []
aris = []
purs = []
best_acc1 = 0
best_acc2 = 0
best_nmi2 = 0
best_ari2 = 0
best_pur2 = 0
best_epoch = 0


T = 1
acc_l = []
nmi_l = []
ari_l = []
loss_l = []

adj = []
adj_feas = []


for i in range(T):
    dataset, dims, view, data_size, class_num = load_data('data/ESRD.mat','data/ESRD.mat')      
        
    alpha = 1
    gamma = 1
    yita = 10
    phi = 0.1
    
    mic_emb_llm = sio.loadmat("data/sim_LLM_ESRD.mat")  # sim_LLM_infant_diabet.mat
    sim_mic = mic_emb_llm['sim_species']
   
    W1, W2, H1, H2, A_, A = JointNMF(dataset, sim_mic, view, class_num, alpha, gamma, yita, phi)
    
    print('alpha:', 'gamma:', 'phi:', 'yita:', alpha, gamma, phi, yita)    
    # sio.savemat('result/lowrep_ESRD_ssmfg.mat', {'W1': W1, 'W2': W2, 'H1': H1, 'H2':H2, 'A_': A_, 'A':A, 'labels': dataset.y}) 
    
    H = (H1 + H2)/2
    
    kmeans = KMeans(n_clusters=class_num, random_state=42)
    kmeans.fit(np.array(H))
    total_label = kmeans.labels_ + 1
    
    label_vector = dataset.y
    label_vector = label_vector.flatten()
    
    acc2, nmi1, ari1, pur1 = evaluate(label_vector, total_label)
    print('ACC{} = {:.4f} NMI{} = {:.4f} ARI{} = {:.4f} pur{}={:.4f}'.format(1, acc2, 1, nmi1, 1, ari1, 1, pur1))

    

