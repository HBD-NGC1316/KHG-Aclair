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
os.system("split -n l/2 -a3 --additional-suffix=.txt ./data/movielens/graph/graph_movie_1step_s.txt ./data/movielens/graph/split_")

## Process only split_xxx.txt files
input_file_list = os.listdir("./data/movielens/graph")
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
item_ent_dic = dict()
item_list = set()
with open("./data/movielens/item_entity/arr_ml2fb_s.txt", "r") as if_file:
    for line in tqdm(if_file):
        line = line.strip().split(maxsplit=2)
        item = line[0]
        entity = line[1]

        item_ent_dic[item] = entity
        item_list.add(int(item))

# Calculate integer size of dividing the element into xx
max_item = max(item_list) ## should contain the largest item number in arr_ml2fb
chunk_size = (max_item // 2) + 1000
# Create a 2D list
list_group_index = [c for c in range(0, 3 * chunk_size, chunk_size)]
print("list_group_index : ", list_group_index)

cnt = 0
ss = set()
for ind in range(len(list_group_index)-1):  
    ## Sort triples for each item + add item-ent
    it_dic = dict()
    for sp_file in tqdm(input_file_list):
        path = "./data/movielens/graph/"
        with open(path+sp_file, "r") as in_file:
            for line2 in in_file:
                line2 = line2.split("\t")
                relation = line2[2]
                #if item in ml2fb: ## item filtering
                item2 = int(line2[0]) ## convert item number to int
                if item2 in item_list:
                    ss.add(item2)
                    if list_group_index[ind] < item2 <= list_group_index[ind+1]:
                        triple = "\t".join(line2[1:])
                        ## Create item:[triple, ..., ] 
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

    
    output_file = "./data/movielens/graph/item_group_"+str(ascii_lowercase[cnt])+".txt"
    print(output_file)
    with open(output_file, "w") as o_f:
        for i in ordered_it:
            item3 = str(i[0])
            itemEnt = item_ent_dic[item3]
            triples = i[1]
            for t in triples:
                o_f.write(itemEnt + "\t" + item3 + "\t" + t)
    cnt += 1
    print("########## Finished Section...", str(list_group_index[ind]), "~", str(list_group_index[ind+1]))

print(len(ss))
