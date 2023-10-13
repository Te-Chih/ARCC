import os
import sys
cur_path=os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, cur_path+"/..")
current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
# import os
os.environ["SETTINGS_MODULE"] = 'settings'
from python_settings import settings
# import settings
import networkx as nx
from buildGraph.utils import *
from node2vec import Node2Vec
import matplotlib.pyplot as plt
from os.path import join
seed = 2021


def IJCNN_getPaperInfo_aminer(paper):

    au_set = set()
    org_set = set()
    ven_set = set()

    # 这里本来就会有一个相同的名字
    for author in paper.get('authors', ''):
        name = author.get('name')
        if len(name) > 0:
            au_set.add(name)
        # 对机构名处理清晰，lower strip  replace(".|-") 用_链接词与词，将机构形成一个整体
        org = clean_name(author.get('org', ''))
        if len(org) > 0:
            org_set.add(org)


    ven_set.add(paper.get('venue', ''))

    return au_set, org_set, ven_set


# 其实是区分的更加清楚了
def IJCNN_generate_adj_matrix_by_rulesim_aminer(paper_ids, paper_infos, paper_dicts=None, idf_Map=None, thr=None):

    paper_num = len(paper_ids)

    cacheMap = {}
    rule_Coau_count = 0
    rule_Coorg_count = 0
    rule_Coven_count = 0
    # rule_label_count = 0

    rule_3Co_matrix = torch.zeros((paper_num, paper_num))
    # rule_Coorg_matrix = torch.zeros((paper_num, paper_num))
    # rule_Coven_matrix = torch.zeros((paper_num, paper_num))
    # rule_label_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i]
        pid1 = paper_id1.split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, ven_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[pid1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, ven_set1 = IJCNN_getPaperInfo_aminer(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, ven_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j]
            pid2 = paper_id2.split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, ven_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[pid2]
                au_set2, org_set2, ven_set2 = IJCNN_getPaperInfo_aminer(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, ven_set2)

            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            mergeVen = len(ven_set1 & ven_set2)  # similaritySet(org_set1, org_set2)  #
            # common_features = set(paper_dicts[pid1]) & set(paper_dicts[pid2])
            # 构造的是无向图
            # 规则1：除同名作者外，有一位相同的作者；
            if mergeAu >= 2:
                rule_Coau_count += 1
                rule_3Co_matrix[i][j] = 1
                rule_3Co_matrix[j][i] = 1
            # 规则2：有一个相同的机构；
            if mergeOrg >= 1:
                rule_Coorg_count += 1
                rule_3Co_matrix[i][j] = 1
                rule_3Co_matrix[j][i] = 1
            # 规则3：有一个相同的venue；
            if mergeVen >= 1:
                rule_Coven_count +=1
                rule_3Co_matrix[i][j] = 1
                rule_3Co_matrix[j][i] = 1

            # 规则：在原有的词库中共现词的idf值超过10
            # 计算共现词的idf值
            # idf_sum = 0
            # for f in common_features:
            #     try:
            #         idf_sum += idf_Map[f]
            #     except:
            #         print(f, j)
            # if idf_sum >= thr:
            #     rule_label_count += 1
            #     rule_label_matrix[i][j] = 1
            #     rule_label_matrix[j][i] = 1
    # rule_3co_matrix = rule_Coven_matrix + rule_Coorg_matrix + rule_Coau_matrix
    # rule_Coau_matrix = rule_Coau_matrix + torch.eye(paper_num, paper_num)
    # diagonal = torch.diag(rule_Coau_matrix)
    # assert torch.all(diagonal == 1)
    # rule_Coorg_matrix = rule_Coorg_matrix + torch.eye(paper_num, paper_num)
    # diagonal = torch.diag(rule_Coorg_matrix)
    # assert torch.all(diagonal == 1)
    # rule_Coven_matrix = rule_Coven_matrix + torch.eye(paper_num, paper_num)
    # diagonal = torch.diag(rule_Coven_matrix)
    # assert torch.all(diagonal == 1)
    # # rule_label_matrix = rule_label_matrix + torch.eye(paper_num, paper_num)
    # # diagonal = torch.diag(rule_label_matrix)
    # assert torch.all(diagonal == 1)



    # print("paper_num{}\tr1:{}\tr2:{}\tr3:{}\tr:{}".format(paper_num,rule_Coau_count/2,rule_Coorg_count/2,rule_Coven_count/2, rule_label_count/2))
    print("paper_num{}\tr1:{}\tr2:{}\tr3:{}".format(paper_num,rule_Coau_count/2,rule_Coorg_count/2,rule_Coven_count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    # return rule_Coau_matrix, rule_Coorg_matrix, rule_Coven_matrix, rule_label_matrix
    return rule_3Co_matrix



def train(paper_feature,data_path,data_name):
    name_papers = parseJson(join(settings.AMINER18_DATA, data_path))      # 论文标签

    # paper_semantic_dict = parseJson(join(settings.AMINER18, 'IJCNN_semantic_dict_TA.json'))
    # idfMap = parseJson(join(settings.AMINER18, 'IJCNN_wordIdf_TA.json'))


    allPaperMatrix_Coau = {}
    # allPaperMatrix_Coorg = {}
    # allPaperMatrix_Coven = {}
    # allPaperMatrix_label = {}
    allPaperRelationEmbedings = {}

    for name_papers in [name_papers]:
        cnt = 0
        for index, name in enumerate(name_papers.keys()):
            try:
                cnt += 1
                # 记录该人名下所有的paper
                papers = []
                truth_pos_count = 0
                for talentid in name_papers[name]:
                    num = len(name_papers[name][talentid])
                    truth_pos_count += (num * (num - 1) / 2)
                    papers.extend(name_papers[name][talentid])

                truthMap = {}
                for talentId in name_papers[name]:
                    for paperId in name_papers[name][talentId]:
                        truthMap[paperId] = talentId

                paper_num = len(papers)
                print(index, name, paper_num)

                # 规则相似度矩阵情况，用规则，构造了同名作者的无向、无权重的图 -> 用邻接矩阵存储；
                # 根据结构规则得到三个邻接矩阵和label矩阵
                rule_3Co_matrix = IJCNN_generate_adj_matrix_by_rulesim_aminer(papers, paper_feature)
                # NOTE:这里都转换为numpy的值了

                rule_sim_a = rule_3Co_matrix.numpy()
                allPaperMatrix_Coau[name] = rule_sim_a.tolist()



                gn = nx.Graph(label=name)
                gn.add_nodes_from(list(range(paper_num)))
                for i in range(paper_num - 1):
                    for j in range(i + 1, paper_num):
                        if rule_3Co_matrix[i][j] > 0:
                            gn.add_edge(i, j, weight=rule_3Co_matrix[i][j])
                            gn.add_edge(j, i, weight=rule_3Co_matrix[i][j])

                n2v = Node2Vec(gn, dimensions=100, walk_length=20, num_walks=10, workers=1)
                model = n2v.fit(window=10, min_count=1, seed=seed)

                paperid2idx = {id: idx for idx, id in enumerate(model.wv.index2word)}
                for idx, id in enumerate(papers):
                    if id in allPaperRelationEmbedings:
                        continue
                    allPaperRelationEmbedings[id] = model.wv.vectors[paperid2idx[str(idx)]].tolist()

            except Exception as e:
                print(e)

    saveJson(join(settings.AMINER18_EMBEDDING, 'RelationEmb_{}_3Co_100.json'.format(data_name)), allPaperRelationEmbedings)

    saveJson(join(settings.AMINER18_DATAPROCESS, '{}_3Co_matrix.json'.format(data_name)), allPaperMatrix_Coau)
    print("finish...")





if __name__ == '__main__':
    paper_feature = parseJson(join(settings.AMINER18_DATA, 'pubs_raw.json'))  # 论文源数据
    for path,name in [('name_to_pubs_test_100.json','test'),('name_to_pubs_train_500.json','train')]:
        train(paper_feature,path,name)
    pass
