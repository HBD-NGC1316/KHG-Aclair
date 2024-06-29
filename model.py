import os
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from dataloader import BasicDataset
import time
import gc
import HGAT

import torch.autograd.profiler as profiler
os.environ['CUDA_LAUNCH_BLOCKING']='1'

class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class KGCL(BasicModel):
    def __init__(self, args, ui_dataset:BasicDataset, hkg_dataset):
        super(KGCL, self).__init__()
        self.config = args
        self.ui_dataset = ui_dataset
        self.hkg_dataset = hkg_dataset

        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.ui_dataset.n_users
        self.num_items = self.ui_dataset.m_items
        ## Hyper graph statistic info 
        self.num_hnodes = self.hkg_dataset.hnodes_count
        self.num_hrelations = self.hkg_dataset.hrelation_count

        print("user:{}, item:{}, hyper nodes:{}".format(
            self.num_users, self.num_items, self.num_hnodes))

        self.latent_dim = self.config.recdim## embedding size ==200
        self.n_layers = self.config.layer
        self.keep_prob = self.config.keepprob## the batch size for bpr loss training procedure=0.8
        self.A_split = False 

        ## Embedding Look-up Table 
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        self.embedding_hnode = torch.nn.Embedding(num_embeddings=self.num_hnodes+1, embedding_dim=self.latent_dim, padding_idx=0)

        self.HGATmodel  = HGAT.DocumentGraph(self.config, self.embedding_hnode).to('cuda:'+str(self.config.gpu_id))
        self.W_R = nn.Parameter(torch.Tensor(
            self.num_hrelations, self.latent_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.W_R, gain=nn.init.calculate_gain('relu'))

        if self.config.pretrain == 0: 
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            nn.init.normal_(self.embedding_hnode.weight, std=0.1)
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config['item_emb']))

        self.f = nn.Sigmoid()

        ## user-item adjency matri
        self.Graph = self.ui_dataset.getSparseGraph()
        self.hkg_dict_ie, self.item2hns = self.hkg_dataset.get_hkg_dict

    def cal_item_embedding_from_hkg(self, hkg_view:dict):

        item_embs = self.embedding_item(torch.LongTensor(list(hkg_view.keys())).to('cuda:'+str(self.config.gpu_id)))
        

        if hkg_view != None:
            ## Hyper GAT Class call
            hgat_emb = HGAT.train_model(self.HGATmodel, item_embs, self.hkg_dataset, hkg_view, self.config) 
            torch.cuda.empty_cache()

        return hgat_emb


    def __dropout_x(self, x, keep_prob):
        
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)

        return g

    
    def __dropout(self, keep_prob):
        ## keep_prob = 0.8
        graph = self.__dropout_x(self.Graph, keep_prob)
        return graph


    def computer(self):
        ## loo-up table, weight
        users_emb = self.embedding_user.weight
        items_emb = self.cal_item_embedding_from_hkg(self.hkg_dict_ie)

        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.config.dropout: ## 0.3
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])

        
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)


        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego
    
    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0,  posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1/2)*(userEmb0.norm(2).pow(2) +
                          posEmb0.norm(2).pow(2) +
                          negEmb0.norm(2).pow(2))/float(len(users))
                          
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        
        # mean or sum
        loss = torch.sum(torch.nn.functional.softplus(-(pos_scores - neg_scores)))
        if (torch.isnan(loss).any().tolist()):
            return None

        return loss, reg_loss


    def view_computer_all(self, g_droped, kg_droped):
            users_emb = self.embedding_user.weight
            items_emb = self.cal_item_embedding_from_hkg(kg_droped)
            all_emb = torch.cat([users_emb, items_emb])
            embs = [all_emb]
            for layer in range(self.n_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
                
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            users, items = torch.split(light_out, [self.num_users, self.num_items])
            return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating
