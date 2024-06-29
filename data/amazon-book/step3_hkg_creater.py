################################################################################################
## Generate a Hypergraph from the 1-step graph.
## (1) Create node sets per item based on relations.
## (2) Create an unencoded hypergraph.
################################################################################################

from tqdm import tqdm
import gc
import os
import re
from string import ascii_lowercase


## Function to write output to file
def write_output(result, c):
    with open("./data/amazon-book/graph/hkg/HKG_ab_"+str(ascii_lowercase[c])+".txt", "w") as out_file:
        for i in result:
            item_ent = i[0]
            item = i[1]
            hkg_li = list(i[2])
            for hkg in hkg_li:
                hyper_e = hkg[0]
                hyper_ns = "\t".join(hkg[1])
                out_file.write(item_ent+"\t"+item+"\t"+hyper_e+"\t"+hyper_ns+"\n")

## Set the target path for outputs
target_parh = "./data/amazon-book/graph"
## Process only item_xxx.txt files
input_file_list = os.listdir(target_parh)
not_split = []
for it in input_file_list:
    regex = re.compile("^item_")
    if regex.search(it) is None:
        not_split.append(it)
for r in not_split:
    input_file_list.remove(r)
sor_input_file_list = sorted(input_file_list)

## Create lists for items and corresponding entities
item_list = []
item_ent_list = []
with open("./data/amazon-book/item_entity/arr_ab2fb_s.txt", "r") as if_file:
    for line in if_file:
        line = line.strip().split("\t")
        item_list.append(line[0])
        item_ent_list.append(line[1])


## Process each item_xxx.txt file in order
item_hkg_list = []
r_pointer = 0 
for graph_file in tqdm(sor_input_file_list): ## item_group file
    print(graph_file)
    ## Get the first and last item numbers from the file
    with open(target_parh+"/"+graph_file, "rb") as f:
        first_line = f.readline().decode()
        try:  # catch OSError in case of a one line file (+)much faster readlines()
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()

        fir_item_id = first_line.strip().split("\t")[1]
        las_item_id = last_line.strip().split("\t")[1]
        ## Get the index values from item_list
        s = item_list.index(fir_item_id)
        e = item_list.index(las_item_id)
        print("fir_item_id: ",fir_item_id)
        print("las_item_id: ",las_item_id)
        print("s: ",s,type(s))
        print("e: ",e, type(e))
    
    
    ## Sort items by relation for each item
    with open(target_parh+"/"+graph_file, "r") as f:
            rel_dic = dict()
            for line in f:
                if s <= e:
                    current_item = item_list[s]
                    current_item_ent = item_ent_list[s]

                    line = line.strip().split("\t")
                    item_ent = line[0]
                    item_id = line[1]
                    relation = line[3]

                    if item_id == current_item:
                        if relation not in rel_dic:
                            rel_dic[relation] = set()
                            rel_dic[relation].add(line[2])
                            rel_dic[relation].add(line[4])
                        else:
                            rel_dic[relation].add(line[2])
                            rel_dic[relation].add(line[4])
                        
                    else:## Create hypergraph list for previous item
                        tmp_list = []
                        tmp_list.append(current_item_ent)
                        tmp_list.append(current_item)
                        hkg = rel_dic.items()
                        tmp_list.append(hkg)
                        item_hkg_list.append(tmp_list)
                        
                        
                        ## Write output every 60,000 tuples for memory efficiency
                        if len(item_hkg_list) == 60000:
                            print("item full!!!")
                            write_output(item_hkg_list, r_pointer)
                            item_hkg_list = []
                            r_pointer+=1

                        ## Reset rel_dic for current new item
                        rel_dic = dict()
                        if relation not in rel_dic:
                            rel_dic[relation] = set()
                            rel_dic[relation].add(line[2])
                            rel_dic[relation].add(line[4])
                        else:
                            rel_dic[relation].add(line[2])
                            rel_dic[relation].add(line[4])


                        s+=1
                else:
                    pass

            ## Process the last item's hypergraph
            tmp_list = []
            tmp_list.append(current_item_ent)
            tmp_list.append(current_item)
            hkg = rel_dic.items()
            tmp_list.append(hkg)
            item_hkg_list.append(tmp_list)
            write_output(item_hkg_list, r_pointer)
            

