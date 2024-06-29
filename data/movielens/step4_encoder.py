######################################################
## Encode User and Item (ratings_s.txt)
## Write in the format: User, item1, ..., item n
## Encode HKG (HKG_ml_a.txt)
######################################################

from tqdm import tqdm
import os
import sys
import re

## Dictionary to store user and item encoding
user_dic = dict() 
item_dic = dict()
## Dictionary to collect items per user
ui_dic = dict()
ss = set()
with open("./data/movielens/user_item/item_enc.txt", "w") as out_item:
    with open("./data/movielens/user_item/ratings_s.txt", "r") as rat_f:
        with open("./data/movielens/user_item/ratings_s_enc.txt", "w") as out:
            user_id = 1
            item_id = 1
            for line in tqdm(rat_f):
                line = line.strip().split("\t")
                user = line[0]
                item = line[1]
                if user not in user_dic:
                    user_dic[user] = user_id
                    user_id += 1
                if item not in item_dic:
                    item_dic[item] = item_id
                    item_id += 1

                if user_dic[user] not in ui_dic:
                    ui_dic[user_dic[user]] = set()
                    ui_dic[user_dic[user]].add(item_dic[item])
                else:
                    ui_dic[user_dic[user]].add(item_dic[item])
            
            for key in ui_dic:
                items = ui_dic[key]
                itemss = list(items)
                ss.update(itemss)
                itemss = "\t".join(map(str,itemss))
                out.write(str(key)+"\t"+itemss+"\n")
    ## Write item number:item encoded number
    for key in item_dic:
        items = item_dic[key]
        out_item.write(key+"\t"+str(items)+"\n")




################ Encode HKG ####################
## Read entity:item id from item-ent file
i2ent_dic = dict()
with open("./data/movielens/item_entity/arr_ml2fb_s.txt", "r") as mlfb_f:
    for line in mlfb_f:
        line = line.strip().split("\t")
        i2ent_dic[line[1]] = line[0]## Actual entity item m.031hcv: original item number

## Read item id:item encode num
i2encode_dic = dict()
max_item = 1
with open("./data/movielens/user_item/item_enc.txt", "r") as rat_encode_f:
    for line2 in rat_encode_f:
        line2 = line2.strip().split("\t")
        i2encode_dic[line2[0]] = line2[1]## Original item number: Encoded item number
        max_item+=1


## Hyeprgraph ori encoding:
target_parh = "./data/movielens/graph/hkg"
## Process only HKG_ml_ori.txt files in target_path
input_file_list = os.listdir(target_parh)
not_split = []
for it in input_file_list:
    regex = re.compile("^HKG_ml_")
    if regex.search(it) is None:
        not_split.append(it)
for r in not_split:
    input_file_list.remove(r)
sor_input_file_list = sorted(input_file_list)

rel_cnt = set()
print("max_item : ", max_item)


tmp_hns_dic = dict()
error_id = set()
rel_dic = dict()
rel_c = 1
for cur_file in sor_input_file_list:
    with open(target_parh+"/enc_"+cur_file, "w") as out_f:
        with open(target_parh+"/"+cur_file, "r") as c_f:
            for line3 in c_f:
                line3 = line3.strip().split("\t")
                item_enity_id = line3[1]
                try:
                    item_encode_ent = i2encode_dic[item_enity_id] ## Original item -> Encoded number
                except Exception as e:
                    error_id.add(item_enity)
                    continue

                ori_item = line3[1]
                hyper_relation = line3[2]
                ## relation encoding
                if hyper_relation not in rel_dic:
                    rel_dic[hyper_relation] = rel_c
                    rel_c+=1
                    hedge_encode = rel_dic[hyper_relation]## hyper edge encoding
                else:
                    hedge_encode = rel_dic[hyper_relation]## hyper edge encoding

                hyper_nodes = line3[3:]

                rel_cnt.add(hyper_relation)
                
                tmp_hns = []
                for node in hyper_nodes:
                    try:
                        node = node[28:-1] ## Remove <http://rdf.freebase.com/ns/, > 
                        if node not in i2ent_dic: ## Not an item entity case
                            if node not in tmp_hns_dic:
                                tmp_hns_dic[node] = max_item
                                tmp_hns.append(max_item)
                                max_item+=1
                            else:
                                tmp_hns.append(tmp_hns_dic[node])
                        else:## Case of item entity
                            tmp_hns.append(i2encode_dic[i2ent_dic[node]]) 

                    except Exception as e:
                        print("Error E:", e)
                
                tmp_hns = "\t".join(map(str, tmp_hns))
                out_f.write(str(item_encode_ent)+"\t*"+ori_item+"\t"+str(hedge_encode)+"\t"+tmp_hns+"\n")





















        

        



        