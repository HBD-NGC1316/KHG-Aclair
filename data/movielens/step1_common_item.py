####################################################################
## Create user_item and item_entity files where items are shared.
## Only retain items common to both user_item and item_entity.
####################################################################
from tqdm import tqdm

myFile = open('./data/movielens/item_entity/arr_ml2fb.txt', 'r') 
myFile_line = myFile.read().count("\n")+1
myFile.close()
line_t = myFile_line *1
print("myFile_line: ",myFile_line)
print("line_t: ",line_t)

with open("./data/movielens/item_entity/arr_ml2fb.txt", "r") as ml_file:
    ml_dic = dict()
    item_list1 = set()
    myFile_cnt = 1
    for line in tqdm(ml_file):
        if myFile_cnt <= line_t:
            line = line.strip().split("\t")
            item_num = line[0]
            item_ent = line[1]
            item_list1.add(item_num)
            ml_dic[item_num] = item_ent
        myFile_cnt +=1    
    

with open("./data/movielens/user_item/ratings100k.txt", "r") as ui_file:
    item_list2 = set()
    for line in tqdm(ui_file):
        line = line.strip().split("\t")
        item_num2 = line[1]
        item_list2.add(item_num2)

## common item
common_item = item_list1.intersection(item_list2) 
print("common_item: ",len(common_item))

it_set=set()
with open('./data/movielens/graph/graph_movie_1step.txt', 'r', encoding='utf-8') as i_file:
    for line in tqdm(i_file):
        line_ = line.strip().split('\t')
        item = line_[0]
        if item in common_item:
            it_set.add(item)


print(len(it_set))
common_item2 = it_set.intersection(common_item)
print("common_item2: ",len(common_item2))

with open('./data/movielens/graph/graph_movie_1step.txt', 'r', encoding='utf-8') as ii_file:
    with open('./data/movielens/graph/graph_movie_1step_s.txt', 'w', encoding='utf-8') as o_file:
        for line in tqdm(ii_file):
            line_ = line.strip().split('\t')
            item = line_[0]
            if item in common_item2:
                it_set.add(item)
                new_line_ = '\t'.join(map(str, line_)) + '\n'
                o_file.write(new_line_)


arr_item_set = set()
with open("./data/movielens/item_entity/arr_ml2fb_s.txt", "w") as ml_out:
    for key in ml_dic:
        if key in common_item2:
            arr_item_set.add(key)
            ml_out.write(key+"\t"+ml_dic[key]+"\n")
print("arr_item_set: ", len(arr_item_set))

rating_item_set = set()
with open("./data/movielens/user_item/ratings_s.txt", "w") as ui_out:
    with open("./data/movielens/user_item/ratings100k.txt", "r") as ui_file:
        for line in ui_file:
            line2 = line.strip().split("\t")
            user = line2[0]
            item_num2 = line2[1]
            if item_num2 in common_item2:
                rating_item_set.add(item_num2)
                ui_out.write(line)
print("rating_item_set: ",len(rating_item_set))

