######################################## 
## UI 데이터가 너무 많아 데이터 셋을 줄이는 코드
## HKG 도 편집함
########################################

from tqdm import tqdm

## 데이터 셋 줄이기
uni_item_set1 = set()
ratio = 1 ## 0~1
## cut한 UI 데이터 셋 아이템 인코딩 (USER는 이미 정렬되어있음)
uni_item_set1 = set()
item_dic1 = dict() ## item: new item enc
cnt = 1
with open("/home/hyejinpark/test/Honey_3_2/data/lastfm/cardi_ratings_s_enc.txt", "r") as ds:
    lines = ds.readlines()
    line_m = int(len(lines)*ratio)
    with open("/home/hyejinpark/test/Honey_3_2/data/lastfm/cardi_ratings_s_enc"+str(ratio*100)+".txt", "w") as output:
        for idx, line in tqdm(enumerate(lines)):
            if idx > line_m:
                pass
            else:
                ## 인코딩
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



## hkg 아이템을 고려하여 엔티티를 인코딩 한다.
## 아이템 가장 큰 값:
edge_dic = dict()
edge_cnt = 1
node_dic = dict()
node_cnt = max(item_dic1.values())+1
uni_item_set = set()
print("node_cnt : ", node_cnt)
with open("/home/hyejinpark/test/Honey_3_2/data/lastfm/hkg/cardi_enc_HKG_lf_aa.txt", "r") as new_hkg:
    with open("/home/hyejinpark/test/Honey_3_2/data/lastfm/hkg/cardi_enc_HKG_lf_aa"+str(ratio*100)+".txt", "w") as new_output:
        ## 아이템 / 릴레이션 / 노드
        for line in tqdm(new_hkg):
            line2 = line.strip().split("\t")
            item = line2[0]
            if item in item_dic1:
                uni_item_set.add(item_dic1[item])
                ## 엣지 인코딩
                edge = line2[2]
                if edge not in edge_dic:
                    edge_dic[edge] = edge_cnt
                    edge = edge_dic[edge]
                    edge_cnt +=1
                else:
                    edge = edge_dic[edge]

                ## nodes 인코딩
                nodes = line2[3:]
                enc_node_list = []
                for node in nodes:
                    ## node 중에 아이템 노드 처리
                    if node == item:
                        enc_node_list.append(str(item_dic1[item]))
                    else:## item이 아닌 노드
                        if node not in node_dic:
                            node_dic[node] = node_cnt
                            enc_node_list.append(str(node_cnt))
                            node_cnt +=1
                        else:
                            enc_node_list.append(str(node_dic[node]))

                strr = "\t".join(enc_node_list)
                new_output.write(str(item_dic1[item])+"\t"+str(edge)+"\t"+strr+"\n")

print("uni_item_set : ", len(uni_item_set))





 



