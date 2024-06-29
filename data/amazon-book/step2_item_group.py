###########################################################################
## (1) 1-hop graph files are not sorted by item order.
## Reorder triples for each item.
## (2) Required for graph encoding... Mapping between items <-> entities with the same meaning.
## Final output format (item_group_xxx.txt): itemEnt  itemID  Head    Relation    Tail
###########################################################################

from tqdm import tqdm
import gc
import os
import re
from string import ascii_lowercase

        
## Split multiple 1-hop graph files
## "split -n l/2 -a3 --additional-suffix=.txt ../1step_graph_07/rel_graph_movie_1step.txt split_"
os.system("split -n l/2 -a3 --additional-suffix=.txt ./data/amazon-book/graph/graph_book_1step_v1.2.txt ./data/amazon-book/graph/split_")

## Process only split_xxx.txt files
input_file_list = os.listdir("./data/amazon-book/graph")
not_split = []
for it in input_file_list:
    regex = re.compile("^split")
    if regex.search(it) is None:
        not_split.append(it)
for r in not_split:
    input_file_list.remove(r)

print("input_file_list : ", input_file_list)


## Create a dictionary mapping items to corresponding Freebase entities
## Create item list and group items
ite_ent_dic = dict()
item_list = set()
with open("./data/amazon-book/item_entity/arr_ab2fb_s.txt", "r") as if_file:
    for line in tqdm(if_file):
        line = line.strip().split(maxsplit = 2)
        item = line[0]
        entity = line[1]

        ite_ent_dic[item] = entity
        item_list.add(str(item))
print("len(item_list)", len(item_list))

cnt = 0
ss = set()
 

it_dic = dict()
for sp_file in tqdm(input_file_list):
    path = "./data/amazon-book/graph/"
    with open(path+sp_file, "r") as in_file:
        for line2 in in_file:
            line2 = line2.split("\t")
            relation = line2[2]
            item2 = line2[0] 
            if item2 in item_list:
                ss.add(item2)
                
                triple = "\t".join(line2[1:])
                ## item:[triple, ..., ] creation
                if item2 not in it_dic:
                    it_dic[item2] = list()
                    it_dic[item2].append(triple)
                else:
                    it_dic[item2].append(triple)


## Ascending sort - based on item
ordered_it = sorted(it_dic.items())


# For memory saving, delete large variable references and call garbage collector()
del it_dic
gc.collect()


output_file = "./data/amazon-book/graph/item_group_"+str(ascii_lowercase[cnt])+".txt"
print(output_file)
with open(output_file, "w") as o_f:
    for i in ordered_it:
        item3 = str(i[0])
        itemEnt = ite_ent_dic[item3]
        triples = i[1]
        for t in triples:
            o_f.write(itemEnt + "\t"+ item3 + "\t" + t)
cnt+=1
print(len(ss))


