import os
import torch
import random
import numpy as np
import networkx as nx
from src.utils import parseJson, saveJson, generate_adj_matrix_by_rulesim
from node2vec import Node2Vec


os.environ['PYTHONHASHSEED'] = '0'
seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.random.manual_seed(seed)


name_papers1 = parseJson('../data/aminerData/name_to_pubs_train_500.json')
name_papers2 = parseJson('../data/aminerData/name_to_pubs_test_100.json')
paper_feature = parseJson('./data/aminerEmbeding/allPapersFeatures.json')


def get_train():
    # 计算训练集结构表征
    allPaperRelationEmbedings = {}
    for index, name in enumerate(name_papers1.keys()):
        try:
            papers = []
            for talentid in name_papers1[name]:
                papers.extend(name_papers1[name][talentid])

            truthMap = {}
            for talentId in name_papers1[name].keys():
                for paperId in name_papers1[name][talentId]:
                    truthMap[paperId] = talentId

            paper_num = len(papers)
            print(index, name, paper_num)

            # 规则相似度矩阵情况
            rule_sim = generate_adj_matrix_by_rulesim(papers, paper_feature)
            rule_con = torch.round(rule_sim)

            gn = nx.Graph(label=name)

            gn.add_nodes_from(list(range(paper_num)))
            for i in range(paper_num - 1):
                for j in range(i+1, paper_num):
                    if rule_con[i][j] > 0:
                        gn.add_edge(i, j, weight=rule_con[i][j])
                        gn.add_edge(j, i, weight=rule_con[i][j])

            n2v = Node2Vec(gn, dimensions=100, walk_length=20, num_walks=10, workers=1)
            model = n2v.fit(window=10, min_count=1, seed=seed)

            paperid2idx = {id: idx for idx, id in enumerate(model.wv.index2word)}
            for idx, id in enumerate(papers):
                if id in allPaperRelationEmbedings:
                    continue
                allPaperRelationEmbedings[id] = model.wv.vectors[paperid2idx[str(idx)]].tolist()

        except Exception as e:
            print(e)

    saveJson("../data/aminerEmbeding/allPaperRelationEmbedings_train_0_100.json", allPaperRelationEmbedings)


def get_test():
    # 计算测试集结构表征
    allPaperRelationEmbedings = {}
    for index, name in enumerate(name_papers2.keys()):
        try:
            papers = []
            for talentid in name_papers2[name]:
                papers.extend(name_papers2[name][talentid])

            truthMap = {}
            for talentId in name_papers2[name].keys():
                for paperId in name_papers2[name][talentId]:
                    truthMap[paperId] = talentId

            paper_num = len(papers)
            print(index, name, paper_num)

            # 规则相似度矩阵情况
            rule_sim = generate_adj_matrix_by_rulesim(papers, paper_feature)
            rule_con = torch.round(rule_sim)

            gn = nx.Graph(name=name)

            gn.add_nodes_from(list(range(paper_num)))
            for i in range(paper_num - 1):
                for j in range(i + 1, paper_num):
                    if rule_con[i][j] > 0:
                        gn.add_edge(i, j, weight=rule_con[i][j])
                        gn.add_edge(j, i, weight=rule_con[i][j])

            n2v = Node2Vec(gn, dimensions=100, walk_length=20, num_walks=10, workers=1)
            model = n2v.fit(window=10, min_count=1, seed=seed)

            paperid2idx = {id: idx for idx, id in enumerate(model.wv.index2word)}
            for idx, id in enumerate(papers):
                if id in allPaperRelationEmbedings:
                    continue
                allPaperRelationEmbedings[id] = model.wv.vectors[paperid2idx[str(idx)]].tolist()
        except Exception as e:
            print(e)

    saveJson("../data/aminerEmbeding/allPaperRelationEmbedings_test_0_100.json", allPaperRelationEmbedings)


if __name__ == '__main__':
    get_train()
    get_test()
    pass
