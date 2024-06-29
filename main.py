
from torch_sparse.tensor import to
import argparse
import torch
from torch import optim
from torch.optim import optimizer, lr_scheduler
from tqdm import tqdm
import time
from datetime import datetime
from torch.utils.data.dataloader import DataLoader
import traceback

from contrast import Contrast
import dataloader
import utils
import model
import Procedure
import gc
import warnings
import os

import torch.autograd.profiler as profiler

warnings.filterwarnings('ignore') 


class Main:
    def __init__(self):
        if args.dataset in ['movielens', 'last-fm', 'amazon-book']:
            self.dataloader = dataloader
            self.hkg_dataset = self.dataloader.HKGDataset(args.dataset)
            self.ui_dataset = self.dataloader.Loader(args.dataset, args.gpu_id)
            
            utils.set_seed(args.seed)
            
            ## Model init()
            torch.cuda.init()
            self.Recmodel = model.KGCL(args, self.ui_dataset, self.hkg_dataset)
            self.Recmodel = self.Recmodel.to('cuda:'+str(args.gpu_id))
            self.contrast_model = Contrast(self.Recmodel, args).to('cuda:'+str(args.gpu_id))
            self.optimizer = optim.Adam(self.Recmodel.parameters(), lr=args.lr)
            self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[1500, 2500], gamma = 0.2)

            self.bpr = utils.BPRLoss(self.Recmodel, self.optimizer, args)
            self.weight_file = "./model_dir/KHGAclair-"+args.dataset+"-"+str(args.recdim)+".pth.tar"


            if args.load: ## '--load', type=int,default=0
                try:
                    self.Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
                    print(f"loaded model weights from {self.weight_file}")
                except FileNotFoundError:
                    print(f"{self.weight_file} not exists, start from beginning")

            
    
    def train(self):
        try:
            # for early stop
            # recall@20
            least_loss = 1e5
            best_result = 0.
            stopping_step = 0
            kgc_joint = True
            result_str=str(args.result)
            t=datetime.now()
            mytime = str(t.month)+"_"+str(t.day) +"_" +str(t.hour)+"_"+str(t.minute)+"_"
            rel_max = args.rel_max
            rel_min = args.rel_min
            with open("./hkg_aclair/result_"+args.dataset+"_"+str(rel_min)+"_"+str(rel_max)+"_"+str(mytime)+".txt", "w") as result_file:
                print("[TEST](train)")
                result_file.write(str(args)+"\n")
                for epoch in tqdm(range(args.epochs), disable=True):
                    start = time.time()
                    self.Recmodel.train()
                    batch_size = args.bpr_batch 
                    print("batch_size:", batch_size)


                    print("[Drop]")
                    if kgc_joint:
                        contrast_views = self.contrast_model.get_views()
                        torch.cuda.empty_cache()
                        gc.collect()

                    print("\n[Joint Learning]")
                    output_information = Procedure.BPR_train_contrast(self.ui_dataset, self.Recmodel, self.bpr, self.contrast_model, contrast_views, epoch, self.optimizer, args)
                    torch.cuda.empty_cache()
                    gc.collect()

                    if epoch<int(args.test_start_epoch): 
                        if (args.dataset == 'amazon-book'): 
                            if  epoch > int(39) and (epoch % 5 != 0):
                                print("[TEST]_epoch<"+str(args.test_start_epoch))
                                result_file.write(str(epoch)+"\n")
                                result = Procedure.Test(self.ui_dataset, self.Recmodel, epoch, result_file, args, args.multicore)
                                result_file.write(str(result)+"\n")
                                print("###########################################")
                                result_file.write("####"+"\n")
                                print(f'[{epoch+1}/{args.epochs}] {output_information}')
                                result_file.write(f'[{epoch+1}/{args.epochs}] {output_information}'+"\n")
                                print("###########################################")
                                result_file.write("\n")

                        if (args.dataset == 'movielens'): 
                            if  epoch > int(9) and (epoch % 5 != 0):
                                print("[TEST]_epoch<"+str(args.test_start_epoch))
                                result_file.write(str(epoch)+"\n")
                                result = Procedure.Test(self.ui_dataset, self.Recmodel, epoch, result_file, args, args.multicore)
                                result_file.write(str(result)+"\n")
                                print("###########################################")
                                result_file.write("####"+"\n")
                                print(f'[{epoch+1}/{args.epochs}] {output_information}')
                                result_file.write(f'[{epoch+1}/{args.epochs}] {output_information}'+"\n")
                                print("###########################################")
                                result_file.write("\n")
                            
                        if epoch % 5 == 0:
                            print("[TEST]_epoch<"+str(args.test_start_epoch))
                            result_file.write(str(epoch)+"\n")
                            result = Procedure.Test(self.ui_dataset, self.Recmodel, epoch, result_file, args, args.multicore)
                            result_file.write(str(result)+"\n")
                            print("###########################################")
                            result_file.write("####"+"\n")
                            print(f'[{epoch+1}/{args.epochs}] {output_information}')
                            result_file.write(f'[{epoch+1}/{args.epochs}] {output_information}'+"\n")
                            print("###########################################")
                            result_file.write("\n")
                    else:
                        if epoch % 1 == 0: ## test_verbose
                            result_file.write("[TEST]_epoch>="+str(args.test_start_epoch)+"\n")
                            result_file.write(str(epoch)+"\n")
                            result = Procedure.Test(self.ui_dataset, self.Recmodel, epoch, result_file,args, args.multicore)
                            result_file.write(str(result)+"\n")
                            result_file.write("###########"+"\n")
                            result_file.write(f'[{epoch+1}/{args.epochs}] {output_information}'+"\n")
                            result_file.write("\n")
                            if result[str(result_str)] > best_result:
                                stopping_step = 0
                                best_result = result[str(result_str)]
                                print("find a better model")
                                torch.save(self.Recmodel.state_dict(), self.weight_file)
                            else:
                                stopping_step += 1
                                    print(f"early stop triggerd at epoch {epoch}")
                                    end_time = datetime.now()
                                    print(f"Program ended at: {end_time}")
                                    result_file.write("early stop triggerd at epoch"+str(epoch)+"\n")
                                    result_file.write("best_result: "+str(best_result)+"\n")
                                    result_file.write("Program ended at:"+str(end_time)+"\n")
                                    break
                    
                    self.scheduler.step()
                    end = time.time()

                    execution_time = end - start
                    print(f"1 Epoch running time: {execution_time} s")
        
        
        except Exception as e:
            print(traceback.format_exc())
            print(e)


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Go HGAT 4 Rec")
    parser.add_argument('--bpr_batch', type=int,default=2048,
                        help="the batch size for bpr loss training procedure 2048")
    parser.add_argument('--recdim', type=int,default=200,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=3,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=float,default=0.3,
                        help="using the dropout or not") 
    parser.add_argument('--alpha', type=float,default=0.3,
                        help="the alpha ") 
    parser.add_argument('--tau', type=float,default=0.2,
                        help="the tau")
    parser.add_argument('--keepprob', type=float,default=0.8,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=12,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='movielens',
                        help="available datasets: [movielens, last-fm, amazon-book]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=0)
    parser.add_argument('--epochs', type=int,default=1000)
    parser.add_argument('--ssl_reg', type=float,default=0.1)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--train', type=int, default=1, help='start train')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--gpu_id', type=int, default='1', help='gpu_id')
    parser.add_argument('--rel_max', type=int, default='1000', help='rel_max')
    parser.add_argument('--rel_min', type=int, default='2', help='rel_min')
    parser.add_argument('--test_start_epoch', type=int, default='100', help='test_start_epoch')
    parser.add_argument('--early_stop_cnt', type=int, default='10', help='early_stop_cnt')
    parser.add_argument('--result', type=str, default="recall", help='result')
    
    args = parser.parse_args()
        
    main = Main()
    if args.train:
        main.train()
    else: 
        main.test()