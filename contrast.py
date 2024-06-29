from cppimport import imp
from numpy import negative, positive
from torch_sparse.tensor import to
from model import KGCL
from random import random, sample
from shutil import make_archive
import torch
import torch.nn as nn
from torch_geometric.utils import degree, to_undirected
import scipy.sparse as sp
import numpy as np
import torch.nn.functional as F
import copy
import time
import torch.autograd.profiler as profiler
import gc


def drop_edge_random(item2hedge, p_drop, padding):
    res = dict()
    
    for item, he in item2hedge.items():
        new_es = list()       
        for e in he:
            rd = random()
            if (rd > p_drop):
                new_es.append(int(e))
            else:
                new_es.append(padding)
        res[item] =new_es

    return res

class Contrast(nn.Module):
    def __init__(self, gcn_model, args):

        super(Contrast, self).__init__()
        self.gcn_model: KGCL = gcn_model 
        self.tau = args.tau
        self.config = args

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        if z1.size()[0] == z2.size()[0]:
            return F.cosine_similarity(z1, z2)
        else:
            z1 = F.normalize(z1)
            z2 = F.normalize(z2)
            return torch.mm(z1, z2.t())
    
    def info_nce_loss_overall(self, z1, z2, z_all):
        def f(x): return torch.exp(x / self.tau)
        positive_pairs = f(self.sim(z1, z2))
        all_sim = f(self.sim(z1, z_all))
        negative_pairs = torch.sum(all_sim, 1)
        loss = torch.sum(-torch.log(positive_pairs / negative_pairs))
        return loss

    def get_kg_views(self):
        hkg = self.gcn_model.hkg_dict_ie ## {item:hyperedge}
        ## random drop
        hkg_p_drop = 0.5
        view1 = drop_edge_random(hkg, hkg_p_drop, self.gcn_model.num_hrelations)
        gc.collect()
        view2 = drop_edge_random(hkg, hkg_p_drop, self.gcn_model.num_hrelations)
        gc.collect()

        return view1, view2
    
    def item_hkg_stability(self, view1, view2):
        hkgv1_ro = self.gcn_model.cal_item_embedding_from_hkg(view1)
        torch.cuda.empty_cache()
        gc.collect()
        
        hkgv2_ro = self.gcn_model.cal_item_embedding_from_hkg(view2)
        torch.cuda.empty_cache()
        gc.collect()
        
        sim = self.sim(hkgv1_ro, hkgv2_ro)
        torch.cuda.empty_cache()
        gc.collect()
        return sim



    def ui_drop_weighted(self, item_mask):
        # item_mask: [item_num]
        item_mask = item_mask.tolist()
        n_nodes = self.gcn_model.num_users + self.gcn_model.num_items
        item_np = self.gcn_model.ui_dataset.trainItem
        gc.collect()
        keep_idx = list()

        for i, j in enumerate(item_np.tolist()):
            if item_mask[j]:
                keep_idx.append(i)
        print(f"finally keep ratio: {len(keep_idx)/len(item_np.tolist()):.2f}")

        keep_idx = np.array(keep_idx)
        user_np = self.gcn_model.ui_dataset.trainUser[keep_idx]
        item_np = item_np[keep_idx]
        ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix(
            (ratings, (user_np, item_np+self.gcn_model.num_users)), shape=(n_nodes, n_nodes))

        
        adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)

        # to coo
        coo = adj_matrix.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        g = torch.sparse.FloatTensor(index, data, torch.Size(
            coo.shape)).coalesce().to('cuda:'+str(self.config.gpu_id))
        g.requires_grad = False
        return g





    def get_ui_views_weighted(self, item_stabilities):
        # kg probability of keep
        item_stabilities = torch.exp(item_stabilities)
        ## normalize
        kg_weights = (item_stabilities - item_stabilities.min()) / \
            (item_stabilities.max() - item_stabilities.min())
        kg_weights = kg_weights.where(
            kg_weights > 0.3, torch.ones_like(kg_weights) * 0.3)

        ui_p_drop = 0.1
        weights = (1-ui_p_drop)/torch.mean(kg_weights)*(kg_weights)
        weights = weights.where(
            weights < 0.95, torch.ones_like(weights) * 0.95)
        item_mask = torch.bernoulli(weights).to(torch.bool)
        # drop
        g_weighted = self.ui_drop_weighted(item_mask)
        g_weighted.requires_grad = False
        
        return g_weighted


    
    def get_views(self, aug_side="both"):
        hkgv1, hkgv2 = self.get_kg_views()
        torch.cuda.empty_cache()
        gc.collect()
        stability = self.item_hkg_stability(hkgv1, hkgv2).to('cuda:'+str(self.config.gpu_id))
        print("stability : ", stability.shape)

        uiv1 = self.get_ui_views_weighted(stability)
        torch.cuda.empty_cache()
        gc.collect()
        uiv2 = self.get_ui_views_weighted(stability)
        torch.cuda.empty_cache()
        gc.collect()

        contrast_views = {
            "kgv1": hkgv1,
            "kgv2": hkgv2,
            "uiv1": uiv1,
            "uiv2": uiv2
        }

        return contrast_views