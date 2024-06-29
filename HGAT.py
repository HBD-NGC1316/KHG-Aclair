import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from sklearn import metrics
import umap
import time
from tqdm import tqdm
from sklearn.random_projection import SparseRandomProjection
import gc
from layers import *
import warnings
import os
from sklearn.decomposition import PCA

import torch.autograd.profiler as profiler

warnings.filterwarnings('ignore') 



class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, alpha, dropout):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, self.alpha, self.dropout,  transfer = False, concat=True)
        
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, self.alpha, self.dropout, transfer = True, concat=False)

    def forward(self, x, H): 
        x = x.unsqueeze(0)
        x = self.gat1(x, H)
        torch.cuda.empty_cache()
        gc.collect()
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gat2(x, H)
        torch.cuda.empty_cache()
        gc.collect()
        return x


class DocumentGraph(Module):
    def __init__(self, args, embedding_hnode):
        super(DocumentGraph, self).__init__()
        self.initial_feature = 200 
        self.hidden_size = 200 
        self.dropout = args.dropout
        ## Embedding Look-up table
        self.embedding = embedding_hnode
        self.hgnn = HGNN_ATT(self.initial_feature, self.initial_feature, self.hidden_size, args.alpha, dropout = self.dropout)

        


    def forward(self, hkg_nodes, HT ):
        hidden = self.embedding(hkg_nodes)
        nodes = self.hgnn(hidden, HT)
        torch.cuda.empty_cache()
        gc.collect()
        return nodes



def train_model(model, item_embs, hkg_dataset, hkg_view, args):
    ## hkg coo
    HT, hkg_nodes = hkg_dataset.generate_hkg_coo(hkg_view)
    hkg_nodes = torch.Tensor(hkg_nodes).long().to('cuda:'+str(args.gpu_id))
    HT = torch.Tensor(HT).float().to('cuda:'+str(args.gpu_id))

    hkg_emb = model(hkg_nodes, HT).squeeze(0)
    torch.cuda.empty_cache()
    gc.collect()
    s_time = time.time()

    items = sorted(hkg_dataset.item2hedges.keys())
    item_node_tensor = torch.empty(len(items), 100, 200).to('cuda:'+str(args.gpu_id))

    for i, inodes in enumerate(hkg_dataset.item_node_set):## 
        max_num = inodes[1]
        nodes = torch.tensor(list(inodes[2:])).to('cuda:'+str(args.gpu_id))
        emb_node = torch.index_select(hkg_emb, 0, nodes).unsqueeze(0).to('cuda:'+str(args.gpu_id))
        emb_mnode = torch.zeros([1, inodes[1], 200]).to('cuda:'+str(args.gpu_id))
        item_node_tensor[i] = torch.cat((emb_node,emb_mnode ), dim = 1)


    ########### Initialize and train the PCA model ########
    n_components = 200
    item_node_tensor_reshaped = item_node_tensor.view(item_node_tensor.shape[0], -1)
    U,S,V = torch.pca_lowrank(item_node_tensor_reshaped, q=n_components, center=True, niter=3)
    del S
    del V
    torch.cuda.empty_cache()
    gc.collect()
    nodes_att_emb = torch.Tensor(U).to('cuda:'+str(args.gpu_id))
  
    return nodes_att_emb





