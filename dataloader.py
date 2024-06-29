from torch.utils.data import Dataset, DataLoader, random_split
import re
import os
import torch
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time
import random

import gc



class HKGDataset(Dataset):
    def __init__(self, path):
        self.dir = "./data/" + path 
        print("Loading the HKG dataset {} ....".format(path))

        self.nhyper_node = 0 
        self.nhyper_edge = 0 
        self.entity_num = 10
        self.HGraph = None
        self.norm_adj = None
        self.item2hedges, self.item2hnodes = self.generate_hkg_data(self.dir)
        self.item_node_set = None
        file_path = os.path.join(self.dir, "item_node_list.npz")
        print("self.item2hedges : ", len(self.item2hedges))
        print("self.item2hnodes : ", len(self.item2hnodes))

        if not os.path.exists(file_path):
            items = sorted(self.item2hedges.keys())
            combined_result = []
            for i, item in enumerate(items):
                combined_node = []
                count_max = 0
                for edge in self.item2hedges[item]:
                    key = str(item)+","+str(edge)
                    node_set = self.item2hnodes[key]

                    # 1의 갯수를 구합니다.
                    count_max += node_set.count(self.nhyper_node)
                    # 1이 아닌 원소를 리스트로 반환합니다.
                    valid_node = [x for x in node_set if x != self.nhyper_node]      
                    combined_node.extend(valid_node)

                combined_list = [item]+ [count_max] + combined_node
                combined_result.append(combined_list)
            self.item_node_set=np.array(combined_result)

            print("^^(self.item_node_set.shape): ",self.item_node_set.shape)     
            np.savez(os.path.join(self.dir, "item_node_list.npz"), arr = self.item_node_set)
            
                 
        else:
            loaded_data = np.load(os.path.join(self.dir, "item_node_list.npz"), allow_pickle = True)
            self.item_node_set = loaded_data["arr"]
            print("^^(self.item_node_set.shape): ",self.item_node_set.shape)


    def generate_hkg_coo(self, hkg_view):
        HT= []
        nodes_list = set()
        if self.HGraph is None:
            try:
                pre_adj_mat = sp.load_npz(self.dir + '/hkg_uH_coo.npz')
                print("successfully loaded...")
                self.norm_adj = pre_adj_mat
            except :
                print("generating adjacency matrix")
                rows = []
                cols = []
                vals = []

                print("self.nhyper_edge : ", self.nhyper_edge)
                print("self.nhyper_node : ", self.nhyper_node)
                for i in hkg_view.keys():
                    edges = hkg_view[i]
                    for edge in edges:
                        if edge == self.nhyper_edge:
                            continue    
                        else:
                            key = str(i)+","+str(edge)

                            for node in self.item2hnodes[key]:
                                if node == self.nhyper_node:
                                    continue
                                else:
                                    rows.append(node)
                                    cols.append(edge)
                                    vals.append(1.0)

                u_H = sp.coo_matrix((vals, (rows, cols)), shape=(self.nhyper_node, self.nhyper_edge))
                sp.save_npz(self.dir + '/hkg_uH_coo.npz', u_H)
                self.norm_adj = u_H
                
        
        HT = self.norm_adj.T.todense()
        self.HGraph = HT

        ## Return a list of nodes in khg corresponding to the items in one batch.
        for item in hkg_view.keys():
            for edge in hkg_view[item]:
                if edge !=  self.nhyper_edge:
                    key = str(item)+","+str(edge)
                    nodes = self.item2hnodes[key]
                    nodes_list.update(nodes)

        nodes_list_fin = list()
        nodes_list_fin.extend(list(nodes_list) + (self.nhyper_node - len(nodes_list)) * [self.nhyper_node])
        return HT, nodes_list_fin




    def generate_hkg_data(self, target_path):
        input_file_list = os.listdir(target_path+"/graph/hkg")
        not_hkg = []
        for it in input_file_list:
            regex = re.compile("^final_HKG") ##for multiple KHG files
            if regex.search(it) is None:
                not_hkg.append(it)
        for r in not_hkg:
            input_file_list.remove(r)
        sor_input_file_list = sorted(input_file_list)
        

        item2hedges_tmp = dict()
        item2hnodes_tmp = dict()
        for ff in sor_input_file_list:
            print("ff:",ff)
            with open(target_path+"/graph/hkg/"+ff) as hkg_file:
                
                for line in hkg_file:
                    line = line.strip().split("\t")
                    ## Reason for - 1:
                    ## The initial dataset starts from 1, so encoding from 0
                    item_enc = int(line[0])-1 
                    hyper_edge = int(line[1])-1
                    tmp_nodes = line[2:]
                    hyper_nodes = [int(x)-1 for x in tmp_nodes]

                    if item_enc not in item2hedges_tmp:
                        item2hedges_tmp[item_enc] = list()
                        item2hedges_tmp[item_enc].append(hyper_edge)
                    else:
                        item2hedges_tmp[item_enc].append(hyper_edge)
                    
                    ## Since the order of the dictionary is not guaranteed, hyper nodes are stored in a dictionary with (item_enc, hyperedge) as keys.
                    nodkey = str(item_enc)+","+str(hyper_edge)

                    if nodkey not in item2hnodes_tmp:
                        item2hnodes_tmp[nodkey] = hyper_nodes
                    else:
                        print(line)


        ## total counts
        hyper_node_set = set()
        for key in item2hnodes_tmp:
            hyper_node_set.update(item2hnodes_tmp[key]) 
        max_node = len(hyper_node_set)
        hyper_edge_set = set()
        for key in item2hedges_tmp:
            hyper_edge_set.update(item2hedges_tmp[key])
        max_edge = len(hyper_edge_set)


        ## item-edge -> DROP 10
        ## item,edge - node  -> DROP 10
        item2hedges = dict()
        item2hnodes = dict()
        for item in item2hedges_tmp:
            if len(item2hedges_tmp[item]) > self.entity_num:
                item2hedges[item] = item2hedges_tmp[item][:self.entity_num]
            else:
                item2hedges[item] = item2hedges_tmp[item]
                item2hedges[item].extend([max_edge]*(self.entity_num-len(item2hedges_tmp[item])))
        del item2hedges_tmp
        gc.collect()


        ## Remove the dropped edge from the node dictionary
        del_list = []
        for key in item2hnodes_tmp:
            n = key.split(",")
            item_n = int(n[0])
            edge_n = int(n[1])

            if edge_n not in item2hedges[item_n]:
                iid = str(item_n)+","+str(edge_n)
                del_list.append(iid)

        for del_k in del_list:
            del item2hnodes_tmp[del_k]

        for itemHEd in item2hnodes_tmp:

            if len(item2hnodes_tmp[itemHEd]) > self.entity_num:
                item2hnodes[itemHEd] = item2hnodes_tmp[itemHEd][:self.entity_num]
            else:
                item2hnodes[itemHEd] = item2hnodes_tmp[itemHEd]
                item2hnodes[itemHEd].extend([max_node]*(self.entity_num-len(item2hnodes_tmp[itemHEd])))
        
        for item, edges in item2hedges.items():
            for edge in edges:
                if edge == max_edge:
                    tmp_ = []
                    key = str(item)+","+str(edge)
                    item2hnodes[key]=[max_node]*self.entity_num

        del item2hnodes_tmp
        gc.collect()

        ## Edge&Node Encoding 
        hyper_edge_set = set()
        for key in item2hedges:
            hyper_edge_set.update(item2hedges[key])
        cur_max_nhyper_edge = len(hyper_edge_set)

        encoding_dic_e = dict()
        encoding_dic_e[max_edge] = cur_max_nhyper_edge
        s_e = 0
        for key in item2hedges:
            edges = item2hedges[key]
            for i, edge in enumerate(edges):
                if edge not in encoding_dic_e:
                    encoding_dic_e[edge] = s_e
                    edges[i] = encoding_dic_e[edge]
                    s_e += 1
                else:
                    edges[i] = encoding_dic_e[edge]
            item2hedges[key] = edges

        hyper_node_set = set()
        for key in item2hnodes:
            hyper_node_set.update(item2hnodes[key]) 
        cur_max_nhyper_node = len(hyper_node_set)
        new_item2hnodes = dict()
        encoding_dic_n = dict()
        encoding_dic_n[max_node] = cur_max_nhyper_node #
        s_e = 0
        for key in item2hnodes:
            item = key.split(",")[0]
            edge = key.split(",")[1]

            new_edge = encoding_dic_e[int(edge)]

            new_key = str(item)+","+str(new_edge)
            nodes = item2hnodes[key]
            
            for i, node in enumerate(nodes):                    
                if node not in encoding_dic_n:
                    encoding_dic_n[node] = s_e
                    nodes[i] = encoding_dic_n[node]
                    s_e += 1
                else:
                    nodes[i] = encoding_dic_n[node]
            new_item2hnodes[new_key] = nodes
 

        ## Final 
        print("[DROP] Unique Hyper Nodes  : ", cur_max_nhyper_node)
        print("[DROP] Unique Hyper Edge : ", cur_max_nhyper_edge)
        self.nhyper_node = cur_max_nhyper_node
        self.nhyper_edge = cur_max_nhyper_edge

        
        return item2hedges, new_item2hnodes

    @property
    def hnodes_count(self):
        return self.nhyper_node
    @property
    def hrelation_count(self):
        return self.nhyper_edge
    
    @property
    def get_hkg_dict(self):
        return self.item2hedges, self.item2hnodes

    
    def __len__(self):
        return len(self.item2hedges)

    def __getitem__(self, idx):
        
        item = idx+1
        return item#, edges




class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")
    
    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def m_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def testDict(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    
    def getUserPosItems(self, users):
        raise NotImplementedError
    
    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError




class Loader(BasicDataset):
    def __init__(self, target_path, gpu_id):

        self.split = False

        ds_name = target_path
        self.dir = "./data/" + target_path 
        self.split_train_test()
        print("Loading the UI dataset {} ....".format(ds_name))
        self.m_item = 0
        self.n_user = 0
        self.traindataSize = 0
        self.testDataSize = 0
        self.gpu_id = gpu_id

        ## For test/train data, generate User and Item arrays
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []

        with open(self.dir+"/train.txt") as train_file:
            ## User / items
            for l in train_file:
                l = l.strip('\n').split('\t')
                ## Reason for - 1:
                ## The initial dataset starts from 1, so encoding from 0
                items = [int(i)-1 for i in l[1:]]
                uid = int(l[0])-1
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items)) 
                trainItem.extend(items)
                self.m_item = max(self.m_item, max(items))
                self.n_user = max(self.n_user, uid)
                self.traindataSize += len(items)
        
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(self.dir+"/test.txt") as test_file:
            for l in test_file:
                l = l.strip('\n').split('\t')

                if l[1]:
                    items = [int(i)-1 for i in l[1:]]
                    uid = int(l[0])-1
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)

        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)
        
        self.m_item += 1
        self.n_user += 1
        self.Graph = None
        print(f"{self.traindataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"{ds_name} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}")

        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        
        self.__testDict = self.__build_test()
        print(f"{ds_name} is ready to go")

    def getUserPosItems(self, users):
        posItems = []
        UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),shape=(self.n_user, self.m_item))
        for user in users:
            posItems.append(UserItemNet[user].nonzero()[1])
        return posItems
    
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getSparseGraph(self):
        print("dataset - Loader")
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.dir + '/s_pre_adj_mat.npz')
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except :
                print("generating batch adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix((self.n_users + self.m_items, self.n_users + self.m_items), dtype=np.float32)


                adj_mat = adj_mat.tolil()## matrix ->list

                ## Make coo
                R = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),shape=(self.n_user, self.m_item)).tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.todok()
                
                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)
                
                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.dir + '/s_pre_adj_mat.npz', norm_adj)

        if self.split == True:
            self.Graph = self._split_A_hat(norm_adj)
        else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to('cuda:'+str(self.gpu_id))
 
        return self.Graph

    
    def _split_A_hat(self,A ):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold*fold_len
            if i_fold == self.folds - 1:
                end =self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(self._convert_sp_mat_to_sp_tensor(A[start:end]).coalesce().to('cuda:'+str(self.gpu_id)))
        return A_fold
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    
    def split_train_test(self):
        if not os.path.exists(self.dir+"/test.txt"):
            ## train/test 8:2분리
            test_ui_dic = dict()
            train_ui_dic = dict()
            with open(self.dir+"/user_item/final_ratings.txt", "r") as ui_file:
                for line in ui_file:
                    line = line.strip().split("\t")
                    user_id = line[0]
                    items = line[1:]
                    
                    ## random pick up
                    percentage = 0.2
                    num_elements_to_select = int(len(items) * percentage)
                    random_elements = random.sample(items, num_elements_to_select)
                    
                    tmp_items = []
                    for i in items:
                        if i not in random_elements:
                            tmp_items.append(i)

                    test_ui_dic[user_id] = random_elements
                    train_ui_dic[user_id] = tmp_items


            with open(self.dir+"/test.txt", "w") as ui_test_file:
                with open(self.dir+"/train.txt", "w") as ui_train_file:
                    for key in test_ui_dic:
                        if len(test_ui_dic[key])>0: 
                            test_items = "\t".join(test_ui_dic[key])
                            ui_test_file.write(key+"\t"+test_items+"\n")
                        train_items = "\t".join(train_ui_dic[key])
                        ui_train_file.write(key+"\t"+train_items+"\n")
    
    
    @property
    def n_users(self):
        return self.n_user
    
    @property
    def m_items(self):
        return self.m_item
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def __len__(self):
        return self.traindataSize

    def __getitem__(self, idx):
        user = self.trainUser[idx]
        pos = random.choice(self._allPos[user])
        while True:
            neg = np.random.randint(0, self.m_item)
            if neg in self._allPos[user]:
                continue
            else:
                break
        return user, pos, neg

