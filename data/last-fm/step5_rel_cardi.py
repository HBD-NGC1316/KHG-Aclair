##########################################################################################
## Preprocess and encode data
## This Python script is designed to preprocess and encode data for a movie recommendation system. It focuses on optimizing data storage and processing efficiency for hypergraphs (HKG) and user-item ratings.
##########################################################################################

import os
import sys
import re
from tqdm import tqdm
import argparse


class Main:
    def run(self):
        ## Perform operations only on enc_HKG_xxx.txt files
        input_file_list = os.listdir("./data/last-fm/graph/hkg")
        not_split = []
        for it in input_file_list:
            regex = re.compile("^enc_HKG")
            if regex.search(it) is None:
                not_split.append(it)
        for r in not_split:
            input_file_list.remove(r)

        rel_dic = dict()

        for f in input_file_list:
            with open("./data/last-fm/graph/hkg/"+f, "r") as input_f:
                for line in tqdm(input_f):
                    line = line.strip().split("\t")
                    rel = line[2]
                    nodes = line[3:]

                    if rel not in rel_dic:
                        rel_dic[rel] = set()
                        rel_dic[rel].update(nodes)
                    else:
                        rel_dic[rel].update(nodes)

        ## Write relation cardinality to rel_cardi.txt
        with open("./data/last-fm/graph/hkg/rel_cardi.txt", "w") as out:
            for k in sorted(rel_dic, key=lambda k: len(rel_dic[k]), reverse=True):
                    out.write("#rel:"+"\t"+str(k)+"\t"+"#cardi:"+"\t"+str(len(rel_dic[k]))+"\n")


        ## Remove relations based on cardinality thresholds
        rel_min = args.rel_min
        rel_max = args.rel_max
        rel_remove = list() ## Relations to be removed
        for k in rel_dic:
            if len(rel_dic[k]) > int(rel_max) or len(rel_dic[k]) < int(rel_min):
                rel_remove.append(k)

        ## Write filtered HKG files without removed relations
        for f in input_file_list:
            with open("./data/last-fm/graph/hkg/cardi_"+f, "w") as output_f:
                with open("./data/last-fm/graph/hkg/"+f, "r") as input_f:
                    for line in tqdm(input_f):
                        line2 = line.strip().split("\t")
                        rel = line2[2]
                        if rel not in rel_remove:
                            output_f.write(line)

        ## Count unique relations and nodes
        rel_set = set()
        node_set = set()
        hkg_item = set()
        for f in input_file_list:
            with open("./data/last-fm/graph/hkg/cardi_"+f, "r") as output_f:
                for line in tqdm(output_f):
                        line2 = line.strip().split("\t")
                        item = line2[0]
                        hkg_item.add(item)
                        rel = line2[2]
                        rel_set.add(rel)
                        nodes = line2[3:]
                        node_set.update(nodes)

        print("Unique rel : ", len(rel_set))
        print("Unique node : ", len(node_set))
        print("Unique hkg_item : ", len(hkg_item))


        ### Remove items from ratings file after deleting relations
        item_list = set()
        with open("./data/last-fm/user_item/ratings_s_enc.txt", "r") as ui_file:
            with open("./data/last-fm/user_item/cardi_ratings_s_enc.txt", "w") as out_file:
                for line in tqdm(ui_file):
                    line2 = line.strip().split("\t")
                    user = line2[0]
                    item_num = line2[1:]
                    itmes = set()
                    for item in item_num:
                        if item in hkg_item:
                            itmes.add(item)
                    if len(itmes)>0:
                        itemss = list(itmes)
                        item_list.update(itemss)
                        itemss = "\t".join(map(str,itemss))
                        out_file.write(str(user)+"\t"+itemss+"\n")

        print("len(item_list): ", len(item_list))


        ## Merge multiple final hypergraph files into one if there are several
        input_file_list = os.listdir("./data/last-fm/graph/hkg")
        not_split = []
        for it in input_file_list:
            regex = re.compile("^cardi_enc_HKG")
            if regex.search(it) is None:
                not_split.append(it)
        for r in not_split:
            input_file_list.remove(r)
        # Sort the file list alphabetically
        sorted_files = sorted([f"./data/last-fm/graph/hkg/{file}" for file in input_file_list])

        # Convert the sorted file list into a space-separated string
        file_list_str = " ".join(sorted_files)

        # Execute shell command to concatenate files
        os.system(f"cat {file_list_str} > ./data/last-fm/graph/hkg/cardi_enc_HKG_lf_aa.txt")



        

        ######################################## 
        ## Start encoding from 0
        ########################################

        ## Reduce the dataset
        uni_item_set1 = set()
        #ratio = 1 ## 0~1
        ## Encode items in the cut UI dataset (USER is already sorted)
        uni_item_set1 = set()
        item_dic1 = dict() ## item: new item enc
        cnt = 1
        with open("./data/last-fm/user_item/cardi_ratings_s_enc.txt", "r") as ds:
            lines = ds.readlines()
            # line_m = int(len(lines)*ratio)
            with open("./data/last-fm/user_item/final_ratings.txt", "w") as output:
                for idx, line in tqdm(enumerate(lines)):
                    ## Encoding
                    encoded_item = set()
                    line2 = line.strip().split("\t")
                    user = line2[0]
                    items = line2[1:]
                    for item in items:
                        if item not in item_dic1:
                            item_dic1[item] = cnt
                            encoded_item.add(str(item_dic1[item]))
                            cnt+=1
                        else:
                            encoded_item.add(str(item_dic1[item]))

                    uni_item_set1.update(encoded_item)
                    strt = "\t".join(encoded_item)
                    output.write(user+"\t"+strt+"\n")

        print("uni_item_set1 : ", len(uni_item_set1))



        ## Encode entities considering HKG items
        ## Maximum item value:
        edge_dic = dict()
        edge_cnt = 1
        node_dic = dict()
        node_cnt = max(item_dic1.values())+1
        uni_item_set = set()
        print("node_cnt : ", node_cnt)
        with open("./data/last-fm/graph/hkg/cardi_enc_HKG_lf_aa.txt", "r") as new_hkg:
            with open("./data/last-fm/graph/hkg/final_HKG.txt", "w") as new_output:
                ## item / relation / node
                for line in tqdm(new_hkg):
                    line2 = line.strip().split("\t")
                    item = line2[0]
                    if item in item_dic1:
                        uni_item_set.add(item_dic1[item])
                        ## Encode edge
                        edge = line2[2]
                        if edge not in edge_dic:
                            edge_dic[edge] = edge_cnt
                            edge = edge_dic[edge]
                            edge_cnt +=1
                        else:
                            edge = edge_dic[edge]

                        ## Encode nodes
                        nodes = line2[3:]
                        enc_node_list = []
                        for node in nodes:
                            ## Handle item nodes
                            if node == item:
                                enc_node_list.append(str(item_dic1[item]))
                            else:## Non-item nodes
                                if node not in node_dic:
                                    node_dic[node] = node_cnt
                                    enc_node_list.append(str(node_cnt))
                                    node_cnt +=1
                                else:
                                    enc_node_list.append(str(node_dic[node]))

                        strr = "\t".join(enc_node_list)
                        new_output.write(str(item_dic1[item])+"\t"+str(edge)+"\t"+strr+"\n")

        print("uni_item_set : ", len(uni_item_set))





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create HKG")
    parser.add_argument('--rel_min', type=str, default='0',
                        help="input the relation cardinality min range. e.g., '0' ")
    parser.add_argument('--rel_max', type=str, default='10000',
                        help="input the relation cardinality max range. e.g., '0' ")

    args = parser.parse_args()

    main = Main()
    main.run()
