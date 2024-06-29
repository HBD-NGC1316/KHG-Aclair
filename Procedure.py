from contrast import Contrast
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from utils import timer
from tqdm import tqdm
import time
import model
import multiprocessing
from sklearn.metrics import roc_auc_score
import torch.autograd.profiler as profiler
import gc


def BPR_train_contrast(dataset, recommend_model, loss_class, contrast_model: Contrast, contrast_views, epoch, optimizer, args): 

    Recmodel: model.KGCL = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class 

    aver_loss = 0.
    aver_loss_main = 0.
    aver_loss_ssl = 0.

    # For SGL
    uiv1, uiv2 = contrast_views["uiv1"], contrast_views["uiv2"]
    kgv1, kgv2 = contrast_views["kgv1"], contrast_views["kgv2"] 
    batch_size = args.bpr_batch 

    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=13)
    b_cnt=1
    for batch_i, train_data in tqdm(enumerate(dataloader), total=len(dataloader), disable=True):

        total_batch = len(dataloader)
        batch_users = train_data[0].long().to('cuda:'+str(args.gpu_id))
        batch_pos = train_data[1].long().to('cuda:'+str(args.gpu_id))
        batch_neg = train_data[2].long().to('cuda:'+str(args.gpu_id))

        l_main = bpr.compute(batch_users, batch_pos, batch_neg) 
        torch.cuda.empty_cache()
        gc.collect()

        items = batch_pos  
        # do SGL:
        # readout
        usersv1_ro, itemsv1_ro = Recmodel.view_computer_all(uiv1, kgv1)
        usersv2_ro, itemsv2_ro = Recmodel.view_computer_all(uiv2, kgv2)
        torch.cuda.empty_cache()
        gc.collect()

        # from SGL source
        items_uiv1 = itemsv1_ro[items]
        items_uiv2 = itemsv2_ro[items]

        l_item = contrast_model.info_nce_loss_overall(items_uiv1, items_uiv2, itemsv2_ro)
        torch.cuda.empty_cache()
        gc.collect()

        users = batch_users
        users_uiv1 = usersv1_ro[users]
        users_uiv2 = usersv2_ro[users]
        l_user = contrast_model.info_nce_loss_overall(users_uiv1, users_uiv2, usersv2_ro)

        del users_uiv1
        del users_uiv2
        del usersv1_ro
        del usersv2_ro
        torch.cuda.empty_cache()
        gc.collect()

        l_ssl = list()
        l_ssl.extend([l_user*args.ssl_reg, l_item*args.ssl_reg])

        del l_user
        del l_item
        del itemsv1_ro
        del itemsv2_ro
        gc.collect()
        torch.cuda.empty_cache()


        if l_ssl:
            l_ssl = torch.stack(l_ssl).sum()
            l_all = l_main+l_ssl
            aver_loss_ssl += l_ssl.cpu().item()
        else:
            l_all = l_main
          

        optimizer.zero_grad()
        l_all.backward()
        optimizer.step()
        aver_loss_main += l_main.cpu().item()
        aver_loss += l_all.cpu().item()
        b_cnt +=1

    
    aver_loss = aver_loss / (total_batch*batch_size)
    aver_loss_main = aver_loss_main / (total_batch*batch_size)
    aver_loss_ssl = aver_loss_ssl / (total_batch*batch_size)
    time_info = timer.dict()
    timer.zero()
    print("##################################################batch - ", batch_i)
    print("aver_loss :", aver_loss)
    print("aver_loss_main:", aver_loss_main)
    print("aver_loss_ssl:", aver_loss_ssl)
    print("############################################################")

    torch.cuda.empty_cache()
    gc.collect()

    return f"loss{aver_loss:.3f} = {aver_loss_ssl:.3f}+{aver_loss_main:.3f}-{time_info}", aver_loss


def test_one_batch(X, topks):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}


def Test(dataset, Recmodel, epoch, result_file, args, multicore=0):
    print("Procedure - Test()")
    
    u_batch_size = args.testbatch
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict ##ui_dataset
    Recmodel: model.KGCL
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    topks = eval(args.topks)
    max_K = max(topks)
    if multicore == 1:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(topks)),
               'recall': np.zeros(len(topks)),
               'ndcg': np.zeros(len(topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users) / 10
        except AssertionError:
            result_file.write("Procedure - Test()\n")
            print(
                f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            result_file.write(
                f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        total_batch = len(users) // u_batch_size + 1
        for batch_users in utils.minibatch(args, users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to('cuda:'+str(args.gpu_id))

            rating = Recmodel.getUsersRating(batch_users_gpu)

            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i] * len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -(1 << 10)
            _, rating_K = torch.topk(rating, k=max_K)
            rating = rating.cpu().numpy()
            del rating
            users_list.append(batch_users)
            rating_list.append(rating_K.cpu())
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, topks))
        scale = float(u_batch_size/len(users))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
      
        if multicore == 1:
            pool.close()
        print(results)
        
        return results