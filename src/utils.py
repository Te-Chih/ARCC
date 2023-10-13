import math
import os
import codecs
import json
import re
from sklearn import metrics
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import numpy as np
import torch
import string
import unicodedata
import scipy.sparse as sp
import torch.nn.functional as F
import yaml
import pandas as pd
from src.clusters import paperClusterByDis
from torch_geometric.data import Data,DataLoader as pygDataLoader
# from clusters import paperClusterByDis
from src.featureUtils import *
# from featureUtils import *
import os
import random
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from bs4 import BeautifulSoup
all_letters = string.ascii_letters

stopwords = {'at', 'based', 'in', 'of', 'for', 'on', 'and', 'to', 'an', 'using', 'with', 'the', 'by', 'we', 'be',
             'is', 'are', 'can', "a", "able", "about", "above", "according", "accordingly", "across", "actually",
             "after", "afterwards", "again", "against", "ain't", "all", "allow", "allows", "almost", "alone", "along",
             "already", "also", "although", "always", "am", "among", "amongst", "an", "and", "another", "any",
             "anybody", "anyhow", "anyone", "anything", "anyway", "anyways", "anywhere", "apart", "appear",
             "appreciate", "appropriate", "are", "aren't", "around", "as", "a's", "aside", "ask", "asking",
             "associated", "at", "available", "away", "awfully", "be", "became", "because", "become", "becomes",
             "becoming", "been", "before", "beforehand", "behind", "being", "believe", "below", "beside", "besides",
             "best", "better", "between", "beyond", "both", "brief", "but", "by", "came", "can", "cannot", "cant",
             "can't", "cause", "causes", "certain", "certainly", "changes", "clearly", "c'mon", "co", "com", "come",
             "comes", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains",
             "corresponding", "could", "couldn't", "course", "c's", "currently", "definitely", "described", "despite",
             "did", "didn't", "different", "do", "does", "doesn't", "doing", "done", "don't", "down", "downwards",
             "during", "each", "edu", "eg", "eight", "either", "else", "elsewhere", "enough", "entirely", "especially",
             "et", "etc", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly",
             "example", "except", "far", "few", "fifth", "first", "five", "followed", "following", "follows", "for",
             "former", "formerly", "forth", "four", "from", "further", "furthermore", "get", "gets", "getting", "given",
             "gives", "go", "goes", "going", "gone", "got", "gotten", "greetings", "had", "hadn't", "happens", "hardly",
             "has", "hasn't", "have", "haven't", "having", "he", "hello", "help", "hence", "her", "here", "hereafter",
             "hereby", "herein", "here's", "hereupon", "hers", "herself", "he's", "hi", "him", "himself", "his",
             "hither", "hopefully", "how", "howbeit", "however", "i'd", "ie", "if", "ignored", "i'll", "i'm",
             "immediate", "in", "inasmuch", "inc", "indeed", "indicate", "indicated", "indicates", "inner", "insofar",
             "instead", "into", "inward", "is", "isn't", "it", "it'd", "it'll", "its", "it's", "itself", "i've", "just",
             "keep", "keeps", "kept", "know", "known", "knows", "last", "lately", "later", "latter", "latterly",
             "least", "less", "lest", "let", "let's", "like", "liked", "likely", "little", "look", "looking", "looks",
             "ltd", "mainly", "many", "may", "maybe", "me", "mean", "meanwhile", "merely", "might", "more", "moreover",
             "most", "mostly", "much", "must", "my", "myself", "name", "namely", "nd", "near", "nearly", "necessary",
             "need", "needs", "neither", "never", "nevertheless", "new", "next", "nine", "no", "nobody", "non", "none",
             "noone", "nor", "normally", "not", "nothing", "novel", "now", "nowhere", "obviously", "of", "off", "often",
             "oh", "ok", "okay", "old", "on", "once", "one", "ones", "only", "onto", "or", "other", "others",
             "otherwise", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "own", "particular",
             "particularly", "per", "perhaps", "placed", "please", "plus", "possible", "presumably", "probably",
             "provides", "que", "quite", "qv", "rather", "rd", "re", "really", "reasonably", "regarding", "regardless",
             "regards", "relatively", "respectively", "right", "said", "same", "saw", "say", "saying", "says", "second",
             "secondly", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible",
             "sent", "serious", "seriously", "seven", "several", "shall", "she", "should", "shouldn't", "since", "six",
             "so", "some", "somebody", "somehow", "someone", "something", "sometime", "sometimes", "somewhat",
             "somewhere", "soon", "sorry", "specified", "specify", "specifying", "still", "sub", "such", "sup", "sure",
             "take", "taken", "tell", "tends", "th", "than", "thank", "thanks", "thanx", "that", "thats", "that's",
             "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby",
             "therefore", "therein", "theres", "there's", "thereupon", "these", "they", "they'd", "they'll", "they're",
             "they've", "think", "third", "this", "thorough", "thoroughly", "those", "though", "three", "through",
             "throughout", "thru", "thus", "to", "together", "too", "took", "toward", "towards", "tried", "tries",
             "truly", "try", "trying", "t's", "twice", "two", "un", "under", "unfortunately", "unless", "unlikely",
             "until", "unto", "up", "upon", "us", "use", "used", "useful", "uses", "using", "usually", "value",
             "various", "very", "via", "viz", "vs", "want", "wants", "was", "wasn't", "way", "we", "we'd", "welcome",
             "well", "we'll", "went", "were", "we're", "weren't", "we've", "what", "whatever", "what's", "when",
             "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "where's", "whereupon",
             "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "who's", "whose",
             "why", "will", "willing", "wish", "with", "within", "without", "wonder", "won't", "would", "wouldn't",
             "yes", "yet", "you", "you'd", "you'll", "your", "you're", "yours", "yourself", "yourselves", "you've",
             "zero", "zt", "ZT", "zz", "ZZ"}


def stat_rm_edge_true_false_num(adj_matrix_tensor,refine_adj_matrix_tensor,label):
    rm_adj_matrix_tensor = refine_adj_matrix_tensor.detach() - adj_matrix_tensor.detach()
    # 删除的边的总数
    rm_edge_num = torch.sum(rm_adj_matrix_tensor == -1)

    # 先得到ground true 的邻接矩阵
    label_adj_matrix = (label.view(-1, 1) == label.view(1, -1)).int().detach()  #
    new_label_adj = label_adj_matrix -  torch.diag_embed(torch.diag(label_adj_matrix)) # 对角线置零

    #
    # label mask,只针对变化的边做判断
    mask = torch.where(rm_adj_matrix_tensor == -1, torch.ones_like(rm_adj_matrix_tensor),
                       torch.zeros_like(rm_adj_matrix_tensor))
    # new_label_adj_mask = mask * new_label_adj

    resule_tensor = rm_adj_matrix_tensor*mask +  new_label_adj*mask

    # 统计-1的个数 即为正确的个数
    true_rm_edge_num = torch.sum(resule_tensor == -1)
    # elements_counter = torch.unique(resule_tensor, return_counts=True)
    # counter_dict=dict(zip(elements_counter[0].tolist(), elements_counter[1].tolist()))
    # true_rm_edge_num = counter_dict[-1]


    ## 总数-正确的个数，即为错误的个数
    false_rm_edge_num = rm_edge_num - true_rm_edge_num
    return int(rm_edge_num.item() / 2), int(true_rm_edge_num.item() / 2), int(false_rm_edge_num.item() / 2)

    #


def stat_add_edge_true_false_num(adj_matrix_tensor,refine_adj_matrix_tensor,label):
    add_adj_matrix_tensor = refine_adj_matrix_tensor.detach() - adj_matrix_tensor.detach()
    ##add edge num
    add_edge_num = torch.sum(add_adj_matrix_tensor == 1)
    # print(add_edge_num / 2)
    # ground truth
    label_adj_matrix = (label.view(-1, 1) == label.view(1, -1)).int().detach()
    diag = torch.diag(label_adj_matrix)
    a_diag = torch.diag_embed(diag)
    new_label_adj = label_adj_matrix - a_diag

    # label mat
    mask = torch.where(add_adj_matrix_tensor == 1, torch.ones_like(add_adj_matrix_tensor),
                       torch.zeros_like(add_adj_matrix_tensor))

    new_label_adj_mask = mask * new_label_adj
    # todo
    # resule_tensor = (new_label_adj_mask == add_adj_matrix_tensor).int() * mask
    resule_tensor = (new_label_adj_mask == add_adj_matrix_tensor* mask ).int() * mask

    ## true edge num
    true_add_edge_num = torch.sum(resule_tensor == 1)
    # print(true_add_edge_num / 2)
    ## false edge num
    false_add_edge_num = add_edge_num - true_add_edge_num
    # print(false_add_edge_num / 2)
    return int(add_edge_num.item() / 2), int(true_add_edge_num.item() / 2), int(false_add_edge_num.item() / 2)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    ).lower()


def parseJson(path):
    try:
        if os.path.exists(path):
            with codecs.open(path, 'r', 'utf-8') as f:
                jsonObj = json.load(f)
            return jsonObj
        else:
            return None
    except Exception as e:
        print(e)
        return None


def saveJson(path, jsonObj):
    try:
        with codecs.open(path, 'w', 'utf-8') as f:
            f.write(json.dumps(jsonObj, ensure_ascii=False, indent=1))

    except Exception as e:
        print(e)


def formatPaperName(originname):
    name = re.sub('\[.+\]', '', originname)
    name = re.sub('\(.+\)', '', name)
    name = re.sub('\{.+\}', '', name)
    name = re.sub('&quot|&gt|&lt|&amp', '', name)
    name = re.sub('[^\u4e00-\u9fa5a-zA-Z,& -]', '', name)

    if len(name) > 0 and (name[-1] == ',' or name[-1] == '&'):
        name = name[:-1]
    return name


def etl(content):
    if content is None:
        return ''
    if isinstance(content, list):
        content = ' '.join(content)

    content = re.sub('&quot|&gt|&lt|&amp', '', content)
    content = re.sub("[\s+\.\!\/,:;$%^*(+\"\')]+|[+——()?【】“”！，。？、~@#￥%……&*（）\|]+", " ", content)
    content = re.sub(r" {2,}", " ", content)
    return content


def formatName(name):
    name = str(name).lower()
    name = re.sub('[ ,-]', '', name)
    name = re.sub('\[.+\]', '', name)
    return name


def is_Chinese(word):
    for ch in word:
        if not ('\u4e00' <= ch <= '\u9fff'):
            return False
    return True


# def cosine_similarity(x, y, norm=False):
#     """ 计算两个向量x和y的余弦相似度 """
#     assert len(x) == len(y)
#     zero_list = [0] * len(x)
#     if x == zero_list or y == zero_list:
#         return float(0)
#     vector_a = np.mat(x)
#     vector_b = np.mat(y)
#     num = float(vector_a * vector_b.T)
#     denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
#     cos = num / denom
#     sim = 0.5 + 0.5 * cos
#     return sim
def get_batch_metric(preds_dict, labels_dict, papers_dict):
    # 拆开 成多个name
    pre_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0

    # tp_sum = 0.0
    # fp_sum = 0.0
    # fn_sum = 0.0
    # tn_sum = 0.0



    for gid in preds_dict.keys():
        pred_t_tensor = torch.stack(preds_dict[gid])
        label_t_tensor = torch.stack(labels_dict[gid])
        cluster_num = len(set(label_t_tensor.tolist()))
        paper_ids = papers_dict[gid]
        sim_matrix = F.cosine_similarity(pred_t_tensor.unsqueeze(1), pred_t_tensor.unsqueeze(0), dim=2).detach()
        dis = 1 - sim_matrix
        result = paperClusterByDis(dis.cpu(), paper_ids, cluster_num, method='AG')
        label2ids_dict = {}
        for i, la in enumerate(label_t_tensor.tolist()):
            label2ids_dict.setdefault(la, []).append(paper_ids[i])
        # 评估指标
        pre_t, recall_t, f1_t = evaluate(result, label2ids_dict)
        # tp_t, fp_t, fn_, tn_t = evaluate_fourMetrics(result, papers_dict)

        pre_sum += pre_t
        recall_sum += recall_t
        f1_sum += f1_t

    precision = pre_sum / len(preds_dict.keys())
    recall = recall_sum / len(preds_dict.keys())
    f1 = f1_sum / len(preds_dict.keys())

    return precision, recall, f1


def __pre_process_df__(data_df):
    name_paperid = {}
    name_label = {}
    for ix, row in data_df.iterrows():
        name = row['name']
        paperid = row['paperid']
        label = row['label']
        if name not in name_paperid:
            name_paperid[name] = []
            name_paperid[name].append(paperid)
        else:
            name_paperid[name].append(paperid)
        if name not in name_label:
            name_label[name] = []
            name_label[name].append(label)
        else:
            name_label[name].append(label)

    return name_paperid, name_label

def transform_pyg_data(name_list,data_df,adj_matrix,paper_embeddings,gcn_node_embeddings):
    dataset = []  # data数据对象的list集合
    name_paperid, name_label = __pre_process_df__(data_df)
    for i in range(len(name_list)):
        # 数据转换
        # 邻接矩阵转换成COO稀疏矩阵及转换
        name = name_list[i]
        edge_index_temp = sp.coo_matrix(adj_matrix[name])
        indices = np.vstack((edge_index_temp.row, edge_index_temp.col))
        edge_index = torch.LongTensor(indices)

        paper_ids = name_paperid[name]
        paper_labels = name_label[name]

        semantic_emb_list = []
        relation_emb_list = []

        for pid in paper_ids:
            semantic_emb_list.append(paper_embeddings[pid.split('-')[0]])
            relation_emb_list.append(gcn_node_embeddings[pid])
            # label_list.append(name_label)

        # semantic节点特征数据转换 [n,100]

        s = torch.FloatTensor(semantic_emb_list)

        # node节点及节点特征数据转换

        x = torch.FloatTensor(relation_emb_list)

        # # 图标签数据转换
        y = torch.LongTensor(paper_labels)

        # # 构建数据集:为一张图，20个节点，每个节点一个特征，Coo稀疏矩阵的边，一个图一个标签
        data = Data(x=x, edge_index=edge_index, y=y, s=s,name=name,pids=paper_ids)  # 构建新型data数据对象
        # print(data)
        dataset.append(data)  # # 将每个data数据对象加入列表
    return dataset


def Aminer_Dataloder(paper_embeddings_path,train_relation_embeddings_path,test_relation_embeddings_path,train_adj_matrix_path,test_adj_matrix_path,train_data_path,eval_data_path,test_data_path,is_train=True,batch_size=1):
    if is_train:
        # emb
        paper_embeddings = parseJson(paper_embeddings_path)
        train_gcn_node_embedings = parseJson(train_relation_embeddings_path)

        # adj
        train_adj_matrix = parseJson(train_adj_matrix_path)
        # eval_adj_matrix = parseJson(eval_adj_matrix_path)

        # df
        train_data_df = pd.read_csv(train_data_path)
        eval_data_df = pd.read_csv(eval_data_path)
        train_name_list = train_data_df['name'].unique().tolist()
        eval_name_list = eval_data_df['name'].unique().tolist()

        # dataset
        train_dataset = transform_pyg_data(train_name_list, train_data_df, train_adj_matrix, paper_embeddings,
                                           train_gcn_node_embedings)
        eval_dataset = transform_pyg_data(eval_name_list, eval_data_df, train_adj_matrix, paper_embeddings,
                                          train_gcn_node_embedings)
        # 生成data loader
        train_loader = pygDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = pygDataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, eval_loader
    else:
        paper_embeddings = parseJson(paper_embeddings_path)
        gcn_node_embeddings = parseJson(test_relation_embeddings_path)

        # adj df
        test_adj_matrix = parseJson(test_adj_matrix_path)
        test_data_df = pd.read_csv(test_data_path)
        test_name_list = test_data_df['name'].unique().tolist()

        test_dataset = transform_pyg_data(test_name_list, test_data_df, test_adj_matrix, paper_embeddings,
                                          gcn_node_embeddings)
        test_loader = pygDataLoader(test_dataset, batch_size=1, shuffle=False)

        return test_loader



def split_mutilGraph_to_SigleGraph(graph_ids,prediction,label,pids):
    graph_ids = graph_ids.tolist()
    prediction = prediction.detach()
    label = label.detach()
    # pids = pids[0]
    preds_dict = {}
    labels_dict = {}
    papers_dict = {}
    j = 0
    gid_set = set()
    gid_set.add(graph_ids[0])
    for i, gid in enumerate(graph_ids):
        if gid not in gid_set:
            j = 0
            gid_set.add(gid)
        preds_dict.setdefault(gid, []).append(prediction[i])
        labels_dict.setdefault(gid, []).append(label[i])
        papers_dict.setdefault(gid, []).append(pids[gid][j])
        j+=1
    return preds_dict,labels_dict,papers_dict

def get_batch_acc(preds_dict, labels_dict, acc_sim_threshold):
    # 拆开 成多个name
    acc_t_sum = 0.0
    for gid in preds_dict.keys():
        pred_t_tensor = torch.stack(preds_dict[gid])
        label_t_tensor = torch.stack(labels_dict[gid])

        sim_matrix = F.cosine_similarity(pred_t_tensor.unsqueeze(1), pred_t_tensor.unsqueeze(0), dim=2).detach()
        pred_matrix = torch.where(sim_matrix > acc_sim_threshold, 1, 0).detach()
        label_matrix = torch.where(label_t_tensor.unsqueeze(1) == label_t_tensor.unsqueeze(1).T, 1, 0).detach()
        check_matrix = torch.where(label_matrix == pred_matrix, 1, 0)
        new_check_matrix = check_matrix - torch.diag_embed(torch.diag(check_matrix))  # 对角线置 0
        acc_t = torch.sum(new_check_matrix).item() / (new_check_matrix.shape[0] * new_check_matrix.shape[1] - new_check_matrix.shape[0])
        acc_t_sum += acc_t
    # acc =
    return acc_t_sum / len(preds_dict.keys())


def evaluate_fourMetrics(preds, truths):
    """
    定义：[(<p1,p2>,pre),label]
    pre = 1 <=> pred[p1] == pred[p2]：预测p1与p2是同一作者
    pre = 0 <=> pred[p1] != pred[p2]：预测p1与p2不是同一作者;
    label = 1 <=> truth[p1] == truth[p2]：真实值p1与p2是同一作者
    label = 0 <=> truth[p1] != truth[p2]：真实值p1与p2不是同一作者
    TP: [(<p1,p2>,1),1], 预测p1与p2是同一作者且预测对了（真实值p1与p2是同一作者）；
    TN: [(<p1,p2>,0),0]，预测p1与p2不是同一作者且预测对了（真实值p1与p2不是同一作者）；
    FP: [(<p1,p2>,1),0]，预测p1与p2是同一作者且预测错了（真实值p1与p2不是同一作者）；
    FN: [(<p1,p2>,0),0]，预测p1与p2不是同一作者且预测错了（真实值p1与p2是同一作者）；
    :param preds:
    :param truths:
    :return:
    """
    predMap = {}
    predList = []

    for i, cluster in enumerate(preds):
        predList.extend(cluster)
        for paperId in cluster:
            predMap[paperId] = i

    truthMap = {}
    for talentId in truths.keys():
        for paperId in truths[talentId]:
            # paperId = str(paperId).split('-')[0]
            truthMap[paperId] = talentId

    # 评价方法调整，方便估计单个元素的类簇
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    n_samples = len(predList)
    counter = 0
    for i in range(n_samples - 1):
        pred_i = predMap[predList[i]]
        for j in range(i + 1, n_samples):
            counter += 1
            pred_j = predMap[predList[j]]
            if pred_i == pred_j:
                if truthMap[predList[i]] == truthMap[predList[j]]:
                    tp += 1
                else:
                    fp += 1
            elif truthMap[predList[i]] == truthMap[predList[j]]:
                fn += 1
            elif truthMap[predList[i]] != truthMap[predList[j]]:
                tn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn

    if tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = tp / tp_plus_fp
        recall = tp / tp_plus_fn
        f1 = (2 * precision * recall) / (precision + recall)

    return tp/counter,fp/counter,fn/counter,tn/counter


def evaluate(preds, truths):
    """
    定义：[(<p1,p2>,pre),label]
    pre = 1 <=> pred[p1] == pred[p2]：预测p1与p2是同一作者
    pre = 0 <=> pred[p1] != pred[p2]：预测p1与p2不是同一作者;
    label = 1 <=> truth[p1] == truth[p2]：真实值p1与p2是同一作者
    label = 0 <=> truth[p1] != truth[p2]：真实值p1与p2不是同一作者
    TP: [(<p1,p2>,1),1], 预测p1与p2是同一作者且预测对了（真实值p1与p2是同一作者）；
    TN: [(<p1,p2>,0),0]，预测p1与p2不是同一作者且预测对了（真实值p1与p2不是同一作者）；
    FP: [(<p1,p2>,1),0]，预测p1与p2是同一作者且预测错了（真实值p1与p2不是同一作者）；
    FN: [(<p1,p2>,0),0]，预测p1与p2不是同一作者且预测错了（真实值p1与p2是同一作者）；
    :param preds:
    :param truths:
    :return:
    """
    predMap = {}
    predList = []

    for i, cluster in enumerate(preds):
        predList.extend(cluster)
        for paperId in cluster:
            predMap[paperId] = i

    truthMap = {}
    for talentId in truths.keys():
        for paperId in truths[talentId]:
            # paperId = str(paperId).split('-')[0]
            truthMap[paperId] = talentId

    # 评价方法调整，方便估计单个元素的类簇
    tp = 0
    fp = 0
    fn = 0
    # tn = 0
    n_samples = len(predList)
    for i in range(n_samples - 1):
        pred_i = predMap[predList[i]]
        for j in range(i + 1, n_samples):
            pred_j = predMap[predList[j]]
            if pred_i == pred_j:
                if truthMap[predList[i]] == truthMap[predList[j]]:
                    tp += 1
                else:
                    fp += 1
            elif truthMap[predList[i]] == truthMap[predList[j]]:
                fn += 1
            # elif truthMap[predList[i]] != truthMap[predList[j]]:
                # tn += 1
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn

    if tp == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = tp / tp_plus_fp
        recall = tp / tp_plus_fn
        f1 = (2 * precision * recall) / (precision + recall)

    return precision, recall, f1

def data_normal_2d(orign_data,dim="col"):
    if dim == "col":
        dim = 1
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    else:
        dim = 0
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data,d_min).true_divide(dst)
    return norm_data


def data_normal_2d_self(orign_data,dim="col"):
    if dim == "col":
        dim = 1
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    else:
        dim = 0
        d_min = torch.min(orign_data,dim=dim)[0]
        for idx,j in enumerate(d_min):
            if j < 0:
                orign_data[idx,:] += torch.abs(d_min[idx])
                d_min = torch.min(orign_data,dim=dim)[0]
    d_max = torch.max(orign_data,dim=dim)[0]
    dst = d_max - d_min
    if d_min.shape[0] == orign_data.shape[0]:
        d_min = d_min.unsqueeze(1)
        dst = dst.unsqueeze(1)
    else:
        d_min = d_min.unsqueeze(0)
        dst = dst.unsqueeze(0)
    norm_data = torch.sub(orign_data,d_min).true_divide(dst)
    return norm_data


def SupCon_Acc_compute(batch_size, data_data_smaper, prediction, label):
    acc_t = []
    for i in range(0, batch_size * data_data_smaper, data_data_smaper):
        prediction_t = prediction[i:i + data_data_smaper, :]
        prediction_t = prediction_t.detach()
        label_t = label[i:i + data_data_smaper]
        label_t = label_t.detach()
        # sim_matrix = F.cosine_similarity(prediction_t.unsqueeze(1), prediction_t.unsqueeze(0), dim=2)
        sim_matrix = F.cosine_similarity(prediction_t.unsqueeze(1), prediction_t.unsqueeze(0), dim=2)

        pred_matrix = torch.where(sim_matrix > 0, 1, 0)
        label_matrix = torch.where(label_t.unsqueeze(1) == label_t.unsqueeze(1).T, 1, 0).detach()
        label_matrix = label_matrix.cpu().detach()
        pred_matrix = pred_matrix.cpu().detach()
        train_metrics_t = torch.sum(torch.where(label_matrix == pred_matrix, 1, 0)).item() / (
                    label_matrix.shape[0] * label_matrix.shape[1])
        acc_t.append(train_metrics_t)
    return acc_t

def SupCon_F1_compute(batch_size, data_data_smaper, prediction, label):
    pres,recs,f1s = [],[],[]
    for i in range(0, batch_size * data_data_smaper, data_data_smaper):
        prediction_t = prediction[i:i + data_data_smaper, :]
        prediction_t = prediction_t.detach()
        label_t = label[i:i + data_data_smaper]
        label_t = label_t.detach()
        # sim_matrix = F.cosine_similarity(prediction_t.unsqueeze(1), prediction_t.unsqueeze(0), dim=2)
        sim_matrix = F.cosine_similarity(prediction_t.unsqueeze(1), prediction_t.unsqueeze(0), dim=2)

        dis = 1 - sim_matrix
        # papers 所有的论文，name_papers[name] 所有的talentid数量，即为聚类数量
        # 聚类算法，
        label_t = label_t.cpu().detach()
        label_t = label_t.data.tolist()
        dis = dis.cpu().detach()
        cluster_num = len(set(label_t))
        papers = range(len(label_t))
        paper_name_dict = {}
        for paperid in papers:
            if label_t[paperid] not in paper_name_dict.keys():
                paper_name_dict[int(label_t[paperid])] = []
                paper_name_dict[int(label_t[paperid])].append(paperid)
            else:
                paper_name_dict[int(label_t[paperid])].append(paperid)
        result = paperClusterByDis(dis, papers, cluster_num, method='AG')
        # 评估指标
        precision, recall, f1 = evaluate(result, paper_name_dict)
        # tp, fp, fn, tn = evaluate_fourMetrics(result, name_papers[name])


        pres.append(precision)
        recs.append(recall)
        f1s.append(f1)
    return pres,recs,f1s

def get_evaluation_conlos(label_matrix, pred_matrix, list_metrics):
    # accuracy 是 预测正确的个数与总数的比值，需要计算预测正确的个数；
    # 首先需要把model预测的相似度或者是每个类的概率，转为类别标签；

    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = torch.sum(torch.where(label_matrix==pred_matrix,1,0)).item() / (label_matrix.shape[0] * label_matrix.shape[1])
    # loss 是数值，是通过loss function计算得到的数值，含义是与真实数值的差距；
    # if 'loss' in list_metrics:
    #     try:
    #         output['loss'] = metrics.log_loss(y_true, y_prob)
    #     except ValueError:
    #         output['loss'] = -1
    # # 混淆矩阵也是计算个数；tp tn fp fn；所以需要类别标签；
    # if 'confusion_matrix' in list_metrics:
    #     output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def get_evaluation(y_true, y_prob, list_metrics):
    # accuracy 是 预测正确的个数与总数的比值，需要计算预测正确的个数；
    # 首先需要把model预测的相似度或者是每个类的概率，转为类别标签；
    if len(y_prob.shape) == 2: # 每个类别的概率值，取最大的概率的类；此处用不上
        y_pred = np.argmax(y_prob, -1)
    else: #将相似度转为二分类类别标签；
        # >= 0.5 预测为1，反之预测为0
        y_pred = [1 if item >= 0.5 else 0 for item in y_prob]
    output = {}
    if 'accuracy' in list_metrics:
        output['accuracy'] = metrics.accuracy_score(y_true, y_pred)
    # loss 是数值，是通过loss function计算得到的数值，含义是与真实数值的差距；
    if 'loss' in list_metrics:
        try:
            output['loss'] = metrics.log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1
    # 混淆矩阵也是计算个数；tp tn fp fn；所以需要类别标签；
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(metrics.confusion_matrix(y_true, y_pred))
    return output


def getPaperInfo(paper):
    """
    用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)

    """
    au_set = set()
    org_set = set()
    common_set = set()

    for word in extract_common_features(paper):
        common_set.add(word)

    for author in paper.get('authors', ''):
        name = author.get('name')
        if len(name) > 0:
            au_set.add(name)
        # 对机构名处理清晰，lower strip  replace(".|-") 用_链接词与词，将机构形成一个整体
        org = clean_name(author.get('org', ''))
        if len(org) > 0:
            org_set.add(org)

    return au_set, org_set, common_set


def getPaperInfo_noidf(paper):
    """
    用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)

    """
    au_set = set()
    org_set = set()
    # common_set = set()
    #
    # for word in extract_common_features(paper):
    #     common_set.add(word)

    for author in paper.get('authors', ''):
        name = author.get('name')
        if len(name) > 0:
            au_set.add(name)
        # 对机构名处理清晰，lower strip  replace(".|-") 用_链接词与词，将机构形成一个整体
        org = clean_name(author.get('org', ''))
        if len(org) > 0:
            org_set.add(org)

    return au_set, org_set


def getPaperInfo_aminer12(paper):
    """
    用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)

    """
    au_set = set()
    org_set = set()
    common_set = set()

    for word in aminer12_extract_common_features(paper):
        common_set.add(word)

    author = paper.get('authors', '')
    if len(author) > 0:
        for name in author.split(","):
            if len(name) > 0:
                au_set.add(name)
        # 对机构名处理清晰，lower strip  replace(".|-") 用_链接词与词，将机构形成一个整体
    # orgs =  paper.get('organization','')
    # if len(orgs) > 0:
    #     for org in orgs.split(","):
    #         org = clean_name(org)
    #         if len(org) > 0:
    #             org_set.add(org)

    orgs = paper.get('organization', '')
    if len(orgs) > 0:
        org = clean_name(orgs)
        if len(org) > 0:
            org_set.add(org)

    return au_set, org_set, common_set


def getPaperFeatures(paperId, cacheMap, paper_features):
    if paperId in cacheMap:
        return cacheMap[paperId]

    au_set = set()
    t_set = set()
    org_set = set()
    venue_set = set()
    for feature in paper_features:
        if '__NAME__' in feature:
            au_set.add(feature)
        elif '__ORG__' in feature:
            org_set.add(feature)
        elif '__VENUE__' in feature:
            venue_set.add(feature)
        else:
            t_set.add(feature)

    cacheMap[paperId] = (au_set, org_set, t_set, venue_set)

    return au_set, org_set, t_set, venue_set


def getPaperFeatures2(paperId, cacheMap, paper_features):
    if paperId in cacheMap:
        return cacheMap[paperId]

    au_set = set()
    t_set = set()
    org_set = set()
    venue_set = set()
    year = ''
    for feature in paper_features:
        if '__NAME__' in feature:
            au_set.add(feature)
        elif '__ORG__' in feature:
            org_set.add(feature)
        elif '__VENUE__' in feature:
            venue_set.add(feature)
        elif '__YEAR__' in feature:
            year = feature[8:]
        else:
            t_set.add(feature)

    cacheMap[paperId] = (au_set, org_set, t_set, venue_set, year)

    return au_set, org_set, t_set, venue_set, year


def similaritySet(strSet1, strSet2):
    mergeCount = 0
    for str1 in strSet1:
        for str2 in strSet2:
            if SequenceMatcher(None, str1, str2).ratio() > 0.9:
                mergeCount += 1
                break
    return mergeCount


def generate_adj_matrix_by_rule(paper_ids, paper_features, threshold=40):
    paper_num = len(paper_ids)
    idfMap = parseJson('./data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    edges = []
    values = []
    for i in range(paper_num - 1):
        paperId1, index1 = paper_ids[i].split('-')
        paper_features1 = set(paper_features[paper_ids[i]])
        # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
        au_set1, org_set1, t_set1 = getPaperInfo(paperId1, int(index1), cacheMap, paper_features1)

        for j in range(i + 1, paper_num):
            paperId2, index2 = paper_ids[j].split('-')
            paper_features2 = set(paper_features[paper_ids[j]])

            common_features = paper_features1 & paper_features2
            idf_sum = 0
            for f in common_features:
                idf_sum += idfMap[f]

            au_set2, org_set2, t_set2 = getPaperInfo(paperId2, int(index2), cacheMap, paper_features2)
            if len(au_set1 & au_set2) >= 2:
                edges.append((i, j))
                values.append(1)
            elif len(au_set1 & au_set2) >= 1 and len(org_set1 & org_set2) >= 1:
                edges.append((i, j))
                values.append(1)
            elif len(au_set1 & au_set2) >= 1 or len(org_set1 & org_set2) >= 1:
                if idf_sum >= threshold:
                    edges.append((i, j))
                    values.append(1)

                elif idf_sum >= int(threshold / 2):
                    edges.append((i, j))
                    values.append(idf_sum / threshold)

    edges = np.array(edges, dtype=np.float32)
    values = np.array(values, dtype=np.float32)

    weight = sp.coo_matrix((values, (edges[:, 0], edges[:, 1])),
                           shape=(len(paper_ids), len(paper_ids)), dtype=np.float32)
    weight = weight + weight.T.multiply(weight.T > weight) - weight.multiply(weight.T > weight)

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(paper_ids), len(paper_ids)), dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, weight


def generate_adj_matrix_by_rule2(paper_ids, paper_features):
    paper_num = len(paper_ids)
    idfMap = parseJson('./data/aminerEmbeding/wordIdf.json')
    cacheMap = {}

    findFather = {}
    label_num = 0
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i]
        paperId1, index1 = paper_id1.split('-')
        if paper_id1 not in findFather:
            findFather[paper_id1] = label_num
            label_num += 1

        paper_features1 = set(paper_features[paper_id1])
        au_set1, org_set1, t_set1 = getPaperInfo(paperId1, int(index1), cacheMap, paper_features1)

        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j]
            paperId2, index2 = paper_id2.split('-')
            paper_features2 = set(paper_features[paper_id2])
            au_set2, org_set2, t_set2 = getPaperInfo(paperId2, int(index2), cacheMap, paper_features2)

            common_features = paper_features1 & paper_features2
            idf_sum = 0
            for f in common_features:
                idf_sum += idfMap[f]

            matched = False
            if len(au_set1 & au_set2) >= 2:
                matched = True
            elif len(au_set1 & au_set2) >= 1 and len(org_set1 & org_set2) >= 1:
                matched = True
            elif len(au_set1 & au_set2) >= 1 or len(org_set1 & org_set2) >= 1:
                if idf_sum >= 8:
                    matched = True

            if matched:
                if paper_id2 in findFather:
                    label = findFather[paper_id1]
                    for id in findFather:
                        if findFather[id] == label:
                            findFather[id] = findFather[paper_id2]
                else:
                    findFather[paper_id2] = findFather[paper_id1]
    print(set(findFather.values()))

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        for j in range(i + 1, paper_num):
            if paper_ids[i] in findFather and paper_ids[j] in findFather and findFather[paper_ids[i]] == findFather[paper_ids[j]]:
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
                continue

    return rule_sim_matrix


def generate_adj_matrix_by_rulesim(paper_ids, paper_infos, threshold=8,idfMap_path = 'and/data/aminerEmbeding/wordIdf.json'):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(idfMap_path)

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1

    # print("paper_num{}\tr1:{}\tr2:{}\tr3:{}\n".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix


def wiw_generate_adj_matrix_by_rulesim(paper_ids, paper_infos, threshold=8,idfMap_path = 'and/data/aminerEmbeding/wordIdf.json'):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(idfMap_path)

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1

    # print("paper_num{}\tr1:{}\tr2:{}\tr3:{}\n".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix



def generate_adj_matrix_by_rulesim_faster(idfMap,paper_ids, paper_infos, MAXGRAPHNUM,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)


    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1

    # print("paper_num{}\tr1:{}\tr2:{}\tr3:{}\n".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix




def generate_adj_matrix_by_rulesim_v2(paper_ids, paper_infos, threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            # elif mergeAu >= 2 and mergeOrg >= 1:
            #     rule2count += 1
            #     rule_sim_matrix[i][j] = 1
            #     rule_sim_matrix[j][i] = 1
            # # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            # elif mergeAu >= 2 or mergeOrg >= 2:
            #     if idf_sum >= threshold:
            #         rule3count +=1
            #         rule_sim_matrix[i][j] = 1
            #         rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1

    print("paper_num{}\tr1:{}\tr2:{}\tr3:{}".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix


def add_edge_by_semantic_similarity():
    pass


def generate_adj_matrix_by_rulesim_and_semantic_deal_gulidian(paper_ids, paper_infos, paper_embedings,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        paper_embeding_list.append(paper_embedings[paper_id1])
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            # elif mergeAu >= 2 and mergeOrg >= 1:
            #     rule2count += 1
            #     rule_sim_matrix[i][j] = 1
            #     rule_sim_matrix[j][i] = 1
            # # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            # elif mergeAu >= 2 or mergeOrg >= 2:
            #     if idf_sum >= threshold:
            #         rule3count +=1
            #         rule_sim_matrix[i][j] = 1
            #         rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1
    # 加入语义信息,旨在解决孤立点
    paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    paper1_embeding = paper_embedings[paper_id1]
    paper_embeding_list.append(paper1_embeding)
    paper_embeding_tensor = torch.tensor(paper_embeding_list)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    _eye = torch.eye(pred_sim.shape[0])
    pred_sim = pred_sim - _eye
    max_index = torch.argmax(pred_sim, dim=1)
    sim_degree = torch.sum(rule_sim_matrix,dim=1)
    # rule4count = 0
    for row in range(0,max_index.shape[0]):
        # cur_paper_id =  paper_ids[row].split('-')[0]
        # 孤立点
        if sim_degree[row].item() <= 0:
            # 找最相似度最高的节点
            col = int(max_index[row].item())
            rule_sim_matrix[row][col] = 1
            rule_sim_matrix[col][row] = 1
                    # add_edge_by_semantic_similarity(paper_embedings,)
                    # cur_paper_emb =paper_embedings[cur_paper_id]



    # print("paper_num{}\tr1:{}\tr2:{}\tr3:{}".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix



def SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, paper_infos, paper_embedings):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    au_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    org_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        paper1_embeding = paper_embedings[paper_id1]
        paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu

            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg


            # 构造的是无向图

    # 最后一篇文章
    paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    paper1_embeding = paper_embedings[paper_id1]
    paper_embeding_list.append(paper1_embeding)

    #list to tensor
    paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float64)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(paper_num)
    ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float64)

    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    sim_matrix = 10*sim_matrix
    return sim_matrix, paper_embeding_list
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1


def fix_GCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos,MAXGRAPHNUM):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    # au_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    # org_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                # rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            # elif mergeAu >= 2 and mergeOrg >= 1:
            #     rule2count += 1
            #     rule_sim_matrix[i][j] = 1
            #     rule_sim_matrix[j][i] = 1
            # # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            # elif mergeAu >= 2 or mergeOrg >= 2:
            #     if idf_sum >= threshold:
            #         rule3count +=1
            #         rule_sim_matrix[i][j] = 1
            #         rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                # rule2count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= 8:
                    # rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    # paper_embeding_list = sem_X1_final
    # zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    # for i in range(paper_num,MAXGRAPHNUM):
    #     paper_embeding_list.append(zero_emb)
    # #list to tensor
    # paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    # pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
    #                                paper_embeding_tensor.unsqueeze(0), dim=2)
    # semantic_sim_matrix = pred_sim - torch.eye(MAXGRAPHNUM)
    # # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)
    #
    # sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    # sim_matrix = sim_matrix.type(torch.float64)
    # sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # # sim_matrix = 10*sim_matrix
    # sim_matrix = sim_matrix.type(torch.float32)
    paper_embeding_list = sem_X1_final
    zero_emb = [0.0 for _ in range(len(paper_embeding_list[0]))]
    for i in range(paper_num, MAXGRAPHNUM):
        paper_embeding_list.append(zero_emb)
    # list to tensor
    paper_embeding_tensor = torch.tensor(paper_embeding_list, dtype=torch.float32)
    return rule_sim_matrix, paper_embeding_tensor











def _GCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos,MAXGRAPHNUM):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    au_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    org_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        # paper1_embeding = paper_embedings[paper_id1]
        # paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu

            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    paper_embeding_list = sem_X1_final
    zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    for i in range(paper_num,MAXGRAPHNUM):
        paper_embeding_list.append(zero_emb)
    #list to tensor
    paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(MAXGRAPHNUM)
    # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    sim_matrix = sim_matrix.type(torch.float64)
    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # sim_matrix = 10*sim_matrix
    sim_matrix = sim_matrix.type(torch.float32)

    return sim_matrix, paper_embeding_tensor
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1














def GCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos,MAXGRAPHNUM):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    au_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    org_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32)
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        # paper1_embeding = paper_embedings[paper_id1]
        # paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu

            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    zero_concat = torch.zeros(MAXGRAPHNUM - paper_num, sem_X1_final.shape[1], dtype=torch.float32).cuda()
    paper_embeding_tensor = torch.cat((sem_X1_final, zero_concat), 0)

    # paper_embeding_list = sem_X1_final
    # zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    # for i in range(paper_num,MAXGRAPHNUM):
    #     paper_embeding_list.append(zero_emb)
    # #list to tensor
    # paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(MAXGRAPHNUM)
    # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    sim_matrix = sim_matrix.type(torch.float64)
    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # sim_matrix = 10*sim_matrix
    sim_matrix = sim_matrix.type(torch.float32)

    return sim_matrix, paper_embeding_tensor
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1




def RecurrentGCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos,MAXGRAPHNUM):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    au_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32).cuda()
    org_rule_sim_matrix = torch.zeros((MAXGRAPHNUM, MAXGRAPHNUM),dtype=torch.float32).cuda()
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        # paper1_embeding = paper_embedings[paper_id1]
        # paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu

            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    zero_concat = torch.zeros(MAXGRAPHNUM - paper_num, sem_X1_final.shape[1], dtype=torch.float32).cuda()
    paper_embeding_tensor = torch.cat((sem_X1_final, zero_concat), 0)

    # paper_embeding_list = sem_X1_final
    # zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    # for i in range(paper_num,MAXGRAPHNUM):
    #     paper_embeding_list.append(zero_emb)
    # #list to tensor
    # paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(MAXGRAPHNUM).cuda()
    # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    sim_matrix = sim_matrix.type(torch.float64)
    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # sim_matrix = 10*sim_matrix
    sim_matrix = sim_matrix.type(torch.float32)

    return sim_matrix.cuda()
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1


def nomax_RecurrentGCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    au_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float32).cuda()
    org_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float32).cuda()
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        # paper1_embeding = paper_embedings[paper_id1]
        # paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu

            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    # zero_concat = torch.zeros(MAXGRAPHNUM - paper_num, sem_X1_final.shape[1], dtype=torch.float32).cuda()
    # paper_embeding_tensor = torch.cat((sem_X1_final, zero_concat), 0)
    paper_embeding_tensor = sem_X1_final
    # paper_embeding_list = sem_X1_final
    # zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    # for i in range(paper_num,MAXGRAPHNUM):
    #     paper_embeding_list.append(zero_emb)
    # #list to tensor
    # paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(paper_num).cuda()
    # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    sim_matrix = sim_matrix.type(torch.float64)
    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # sim_matrix = 10*sim_matrix
    sim_matrix = sim_matrix.type(torch.float32)

    return sim_matrix.cuda()
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1



def nomax_impGCN_SemanticAndRule_Similarity_Guide_Build_Graph(paper_ids, sem_X1_final,paper_infos):
    # print("执行SemanticAndRule_Similarity_Guide_Build_Graph")
    paper_num = len(paper_ids)
    # current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
    # idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')
    cacheMap = {}
    # rule1count = 0
    # rule2count = 0
    # rule3count = 0
    adj_matrix = torch.zeros((paper_num, paper_num),dtype=torch.int64).cuda()
    au_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float32).cuda()
    org_rule_sim_matrix = torch.zeros((paper_num, paper_num),dtype=torch.float32).cuda()
    # semantic_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        # paper1_embeding = paper_embedings[paper_id1]
        # paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1 = getPaperInfo_noidf(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2 = getPaperInfo_noidf(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2)



            # 计算两篇文章作者 机构相似度

            sumAu = len(au_set1 | au_set2) - 1
            sumOrg = len(org_set1 | org_set2)

            mergeAu = len(au_set1 & au_set2) - 1  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2) # similaritySet(org_set1, org_set2)  #

            if sumAu > 0  and mergeAu > 0 :
                simAu = math.tanh(3*mergeAu/sumAu)
                au_rule_sim_matrix[i][j] = simAu
                au_rule_sim_matrix[j][i] = simAu
                # adj_matrix[i][j] = 1
                # adj_matrix[i][j] = 1
            if  sumOrg > 0 and mergeOrg > 0 :
                simOrg = math.tanh(3 * mergeOrg / sumOrg)
                org_rule_sim_matrix[i][j] = simOrg
                org_rule_sim_matrix[j][i] = simOrg
                # adj_matrix[i][j] = 1
                # adj_matrix[i][j] = 1


            # 构造的是无向图

    # 最后一篇文章
    # paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    # paper1_embeding = paper_embedings[paper_id1]
    # paper_embeding_list.append(paper1_embeding)
    # zero_concat = torch.zeros(MAXGRAPHNUM - paper_num, sem_X1_final.shape[1], dtype=torch.float32).cuda()
    # paper_embeding_tensor = torch.cat((sem_X1_final, zero_concat), 0)
    paper_embeding_tensor = sem_X1_final
    # paper_embeding_list = sem_X1_final
    # zero_emb = [0.0 for i in range(len(paper_embeding_list[0])) ]
    # for i in range(paper_num,MAXGRAPHNUM):
    #     paper_embeding_list.append(zero_emb)
    # #list to tensor
    # paper_embeding_tensor = torch.tensor(paper_embeding_list,dtype=torch.float32)
    pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
                                   paper_embeding_tensor.unsqueeze(0), dim=2)
    semantic_sim_matrix = pred_sim - torch.eye(paper_num).cuda()
    # ze_ = torch.zeros((paper_num, paper_num),dtype=torch.float64)
    # semantic_sim_matrix = torch.where(semantic_sim_matrix > 0.5, semantic_sim_matrix, ze_)

    sim_matrix  =  (au_rule_sim_matrix + org_rule_sim_matrix + semantic_sim_matrix) / torch.tensor(3.0, dtype=torch.float32)
    sim_matrix = sim_matrix.type(torch.float64)
    sim_matrix = torch.where(sim_matrix < 0.05, 0.0, sim_matrix)
    # sim_matrix = 10*sim_matrix
    sim_matrix = sim_matrix.type(torch.float32)

    return sim_matrix.cuda()
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix, dim=1)
    # # rule4count = 0
    # for row in range(0, max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #         col = int(max_index[row].item())
    #         rule_sim_matrix[row][col] = 1
    #         rule_sim_matrix[col][row] = 1





def initial_generate_adj_matrix_by_semantic_guide_graph(paper_ids, paper_infos, paper_embedings,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0


    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        paper1_embeding = paper_embedings[paper_id1]
        paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            paper2_embeding = paper_embedings[paper_id2]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图

            # 计算两个节点的相似度
            cosins_sim_score = F.cosine_similarity(torch.tensor(paper1_embeding, dtype=torch.float32), torch.tensor(paper2_embeding, dtype=torch.float32),
                                             dim=0).item()
            cosins_sim_score *= 10
            if cosins_sim_score > 0:
                # 规则1：除同名作者外，有两位相同的作者；
                if mergeAu >= 3:
                    rule1count += 1

                    rule_sim_matrix[i][j] =  mergeAu-1 + cosins_sim_score
                    rule_sim_matrix[j][i] =  mergeAu-1 + cosins_sim_score


                # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
                elif mergeAu >= 2 and mergeOrg >= 1:
                    rule2count += 1
                    rule_sim_matrix[i][j] = (mergeAu-1 + mergeOrg) + cosins_sim_score
                    rule_sim_matrix[j][i] = (mergeAu-1 + mergeOrg) + cosins_sim_score
                # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
                elif mergeAu >= 2 or mergeOrg >= 1:
                    if idf_sum >= threshold:
                        rule3count +=1
                        if mergeAu >= 2:
                            rule_sim_matrix[i][j] = (mergeAu - 1) + cosins_sim_score
                            rule_sim_matrix[j][i] =  (mergeAu - 1)  +  cosins_sim_score
                        else:
                            rule_sim_matrix[i][j] = mergeOrg  +  cosins_sim_score
                            rule_sim_matrix[j][i] = mergeOrg  +  cosins_sim_score
                # 规则4， 利用语义添加高置信度边
                elif cosins_sim_score > 5: #这个可以调节
                      # 添加高置信度的边
                     rule_sim_matrix[i][j] = cosins_sim_score
                     rule_sim_matrix[j][i] = cosins_sim_score

            else:
                pass
    paper_id1 = paper_ids[paper_num - 1].split('-')[0]
    paper1_embeding = paper_embedings[paper_id1]
    paper_embeding_list.append(paper1_embeding)
    return rule_sim_matrix, paper_embeding_list





def initial_generate_adj_matrix(paper_ids, paper_infos, paper_embedings,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        paper1_embeding = paper_embedings[paper_id1]
        paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            paper2_embeding = paper_embedings[paper_id2]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图

            # 计算两个节点的相似度
            cosins_sim_score = F.cosine_similarity(torch.tensor(paper1_embeding, dtype=torch.float32), torch.tensor(paper2_embeding, dtype=torch.float32),
                                             dim=0).item()
            cosins_sim_score *= 10

            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1

                rule_sim_matrix[i][j] =  1
                rule_sim_matrix[j][i] =  1


            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule2count += 1
                rule_sim_matrix[i][j] =1
                rule_sim_matrix[j][i] =1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    if mergeAu >= 2:
                        rule_sim_matrix[i][j] = 1
                        rule_sim_matrix[j][i] =  1
                    else:
                        rule_sim_matrix[i][j] = 1
                        rule_sim_matrix[j][i] = 1
                # 规则4， 利用语义添加高置信度边
                # elif cosins_sim_score > 5: #这个可以调节
                #       # 添加高置信度的边
                #      rule_sim_matrix[i][j] = cosins_sim_score
                #      rule_sim_matrix[j][i] = cosins_sim_score
    paper_id1 = paper_ids[paper_num-1].split('-')[0]
    paper1_embeding = paper_embedings[paper_id1]
    paper_embeding_list.append(paper1_embeding)
    # 加入语义信息,旨在解决孤立点

    # paper_embeding_tensor = torch.tensor(paper_embeding_list)
    # pred_sim = F.cosine_similarity(paper_embeding_tensor.unsqueeze(1),
    #                                paper_embeding_tensor.unsqueeze(0), dim=2)
    # _eye = torch.eye(pred_sim.shape[0])
    # pred_sim = pred_sim - _eye
    # max_index = torch.argmax(pred_sim, dim=1)
    # sim_degree = torch.sum(rule_sim_matrix,dim=1)
    # # rule4count = 0
    # for row in range(0,max_index.shape[0]):
    #     # cur_paper_id =  paper_ids[row].split('-')[0]
    #     # 孤立点
    #     if sim_degree[row].item() <= 0:
    #         # 找最相似度最高的节点
    #
    #         col = int(max_index[row].item())
    #         sim_score=pred_sim[row][col].item() * 10
    #         if sim_score > 2.5: #为孤立点添加一条边。
    #             rule_sim_matrix[row][col] = sim_score
    #             rule_sim_matrix[col][row] = sim_score




    # print("paper_num{}\tr1:{}\tr2:{}\tr3:{}".format(paper_num,rule1count/2,rule2count/2,rule3count/2))
    # with open(file="ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("r1:{}\tr2:{}\tr3:{}\n".format(rule1count,rule2count,rule3count))
        # fileTmp.flush()
    return rule_sim_matrix, paper_embeding_list


def generate_adj_matrix_by_semantic_guide_graph(paper_ids, paper_infos, paper_embedings,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)
    current_path = os.path.abspath(os.path.dirname(__file__)) + '/'

    idfMap = parseJson(current_path+'../data/aminerEmbeding/wordIdf.json')

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    paper_embeding_list = []
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        paper1_embeding = paper_embedings[paper_id1]
        paper_embeding_list.append(paper1_embeding)
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            paper2_embeding = paper_embedings[paper_id2]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                try:
                    idf_sum += idfMap[f]
                except:
                    print(f,j)
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图

            # 计算两个节点的相似度
            cosins_sim_score = F.cosine_similarity(torch.tensor(paper1_embeding, dtype=torch.float32), torch.tensor(paper2_embeding, dtype=torch.float32),
                                             dim=0).item()
            cosins_sim_score *= 10
            if cosins_sim_score > 0:
                # 规则1：除同名作者外，有两位相同的作者；
                if mergeAu >= 3:
                    rule1count += 1

                    rule_sim_matrix[i][j] =  mergeAu-1 + cosins_sim_score
                    rule_sim_matrix[j][i] =  mergeAu-1 + cosins_sim_score


                # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
                elif mergeAu >= 2 and mergeOrg >= 1:
                    rule2count += 1
                    rule_sim_matrix[i][j] = (mergeAu-1 + mergeOrg) + cosins_sim_score
                    rule_sim_matrix[j][i] = (mergeAu-1 + mergeOrg) + cosins_sim_score
                # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
                elif mergeAu >= 2 or mergeOrg >= 1:
                    if idf_sum >= threshold:
                        rule3count +=1
                        if mergeAu >= 2:
                            rule_sim_matrix[i][j] = (mergeAu - 1) + cosins_sim_score
                            rule_sim_matrix[j][i] =  (mergeAu - 1)  +  cosins_sim_score
                        else:
                            rule_sim_matrix[i][j] = mergeOrg  +  cosins_sim_score
                            rule_sim_matrix[j][i] = mergeOrg  +  cosins_sim_score
                # 规则4， 利用语义添加高置信度边
                elif cosins_sim_score > 5: #这个可以调节
                      # 添加高置信度的边
                     rule_sim_matrix[i][j] = cosins_sim_score
                     rule_sim_matrix[j][i] = cosins_sim_score

            else:
                pass

    return rule_sim_matrix


def generate_adj_matrix_by_rulesim_aminer12(paper_ids, paper_infos, idfMap,threshold=8):
    # print("执行generate_adj_matrix_by_rulesim")
    paper_num = len(paper_ids)

    cacheMap = {}
    rule1count = 0
    rule2count = 0
    rule3count = 0
    # rule = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i].split('-')[0]
        if paper_id1 in cacheMap:
            au_set1, org_set1, common_set1 = cacheMap[paper_id1]
        else:
            paper_info1 = paper_infos[paper_id1]
            # 用一个论文原始数据，得到三个集合；1.（title/abs/venue的clean词集合）2.(处理后的org集合) 3.(name集合)
            au_set1, org_set1, common_set1 = getPaperInfo_aminer12(paper_info1)
            # 将三个集合存储起来
            cacheMap[paper_id1] = (au_set1, org_set1, common_set1)
        # 与其余文章，判断是否是有边；
        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j].split('-')[0]
            if paper_id2 in cacheMap:
                au_set2, org_set2, common_set2 = cacheMap[paper_id2]
            else:
                paper_info2 = paper_infos[paper_id2]
                au_set2, org_set2, common_set2 = getPaperInfo_aminer12(paper_info2)
                cacheMap[paper_id2] = (au_set2, org_set2, common_set2)

            common_features = common_set1 & common_set2  # 去paper中词干的交集
            idf_sum = 0
            # 用同词干的idf值的和 去判断；
            for f in common_features:
                idf_sum += idfMap[f]
            # 找出同作者以及同机构的数量
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            # 构造的是无向图
            # 规则1：除同名作者外，有两位相同的作者；
            if mergeAu >= 3:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则2：除同名作者外，有一位相同的作者+有同一个机构；
            elif mergeAu >= 2 and mergeOrg >= 1:
                rule1count += 1
                rule_sim_matrix[i][j] = 1
                rule_sim_matrix[j][i] = 1
            # 规则3：重复词干的idf>8 and  (除同名作者外有一个相同的作者 or  有一个相同的机构)；
            elif mergeAu >= 2 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule3count +=1
                    rule_sim_matrix[i][j] = 1
                    rule_sim_matrix[j][i] = 1

    # print("{}\t{}\t{}\n".format(rule1count,rule2count,rule3count))
    # with open(file="./tmp/ruleAnalysis.txt", mode="a", encoding="utf-8") as fileTmp:
    #     # fileTmp.write("start...")
    #     fileTmp.write("{}\t{}\t{}\n".format(rule1count,rule2count,rule3count))
    #     fileTmp.flush()
    return rule_sim_matrix





def generate_adj_matrix_by_rulesim2(paper_ids, paper_features, threshold=20):
    paper_num = len(paper_ids)
    idfMap = parseJson('./data/aminerEmbeding/wordIdf.json')
    cacheMap = {}

    rule_sim_matrix = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i]
        paperId1, index1 = paper_id1.split('-')
        paper_features1 = set(paper_features[paper_id1])
        au_set1, org_set1, t_set1 = getPaperInfo(paperId1, int(index1), cacheMap, paper_features1)

        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j]
            paperId2, index2 = paper_id2.split('-')
            paper_features2 = set(paper_features[paper_id2])
            common_features = paper_features1 & paper_features2
            idf_sum = 0
            for f in common_features:
                idf_sum += idfMap[f]

            au_set2, org_set2, t_set2 = getPaperInfo(paperId2, int(index2), cacheMap, paper_features2)
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            if mergeAu >= 2:
                rule_sim_matrix[i][j] += 1
                rule_sim_matrix[j][i] += 1

            if mergeAu >= 1 and mergeOrg >= 1:
                rule_sim_matrix[i][j] += 1
                rule_sim_matrix[j][i] += 1

            if mergeAu >= 1 or mergeOrg >= 1:
                if idf_sum >= threshold:
                    rule_sim_matrix[i][j] += 1
                    rule_sim_matrix[j][i] += 1
                else:
                    rule_sim_matrix[i][j] += idf_sum / threshold
                    rule_sim_matrix[j][i] += idf_sum / threshold

    return rule_sim_matrix


def generate_muti_adj_matrix(paper_ids, paper_features, threshold=40):
    paper_num = len(paper_ids)
    idfMap = parseJson('./data/aminerEmbeding/wordIdf.json')
    cacheMap = {}

    rule_sim_matrix1 = torch.zeros((paper_num, paper_num))
    rule_sim_matrix2 = torch.zeros((paper_num, paper_num))
    rule_sim_matrix3 = torch.zeros((paper_num, paper_num))
    for i in range(paper_num - 1):
        paper_id1 = paper_ids[i]
        paperId1, index1 = paper_id1.split('-')
        paper_features1 = set(paper_features[paper_id1])
        au_set1, org_set1, t_set1 = getPaperInfo(paperId1, int(index1), cacheMap, paper_features1)

        for j in range(i + 1, paper_num):
            paper_id2 = paper_ids[j]
            paperId2, index2 = paper_id2.split('-')
            paper_features2 = set(paper_features[paper_id2])
            common_features = paper_features1 & paper_features2
            idf_sum = 0
            for f in common_features:
                idf_sum += idfMap[f]

            au_set2, org_set2, t_set2 = getPaperInfo(paperId2, int(index2), cacheMap, paper_features2)
            mergeAu = len(au_set1 & au_set2)  # similaritySet(au_set1, au_set2)  #
            mergeOrg = len(org_set1 & org_set2)  # similaritySet(org_set1, org_set2)  #
            if mergeAu >= 2:
                rule_sim_matrix1[i][j] = 1
                rule_sim_matrix1[j][i] = 1

            if mergeAu >= 1 and mergeOrg >= 1:
                rule_sim_matrix2[i][j] = 1
                rule_sim_matrix2[j][i] = 1

            if idf_sum >= 20:
                rule_sim_matrix3[i][j] = 1
                rule_sim_matrix3[j][i] = 1

    return rule_sim_matrix1, rule_sim_matrix2, rule_sim_matrix3


def generate_samples(pos_matrix, pred_matrix, cosine_matrix, sample_num=10):
    sample_papers = []
    sample_pairwises = []
    labels = []

    for i in range(pos_matrix.shape[0]):
        try:
            neighboors = []
            noneighboors = []
            for j in range(pos_matrix.shape[1]):
                if i != j:
                    if pos_matrix[i][j] == 1:
                        neighboors.append(j)
                    else:
                        noneighboors.append(j)

            if len(neighboors) == 0:
                continue

            # 正样本的采样方式，依照预测的相似度进行采样，预测的相似度越大越容易被采样
            pos_pro = pred_matrix[i, neighboors]
            pos_pro = F.softmax(pos_pro, dim=0).numpy()

            # 负样本的采样方式，依照样本之间的相似度进行采样，样本之间相似度越大越容易被负采样
            neg_pro = cosine_matrix[i, noneighboors]
            neg_pro = F.softmax(neg_pro, dim=0).numpy()

            poss = np.random.choice(neighboors, size=sample_num, p=pos_pro)
            sample_papers.extend([i] * sample_num)
            sample_pairwises.extend(poss)
            labels.extend([1] * sample_num)

            negs = np.random.choice(noneighboors, size=sample_num, p=neg_pro)
            sample_papers.extend([i] * sample_num)
            sample_pairwises.extend(negs)
            labels.extend([0] * sample_num)
        except Exception as e:
            print(e)

    return sample_papers, sample_pairwises, labels


def generate_adj_matrix_by_threshold(paper_ids, paper_features, threshold=40):
    edges = []
    idfMap = parseJson('./data/aminerEmbeding/wordIdf.json')
    for i in range(len(paper_ids)):
        paper_features1 = set(paper_features[paper_ids[i]])

        for j in range(i + 1, len(paper_ids)):
            paper_features2 = set(paper_features[paper_ids[j]])
            common_features = paper_features1 & paper_features2
            idf_sum = 0
            for f in common_features:
                idf_sum += idfMap[f] if f in idfMap else 0

            if idf_sum >= threshold:
                edges.append([i, j])

    edges = np.array(edges, dtype=np.float32)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(paper_ids), len(paper_ids)), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, adj


def generate_adj_matrix_by_discriminator(emb, dis_model):
    emb = dis_model(emb)
    emb = torch.tensor(emb.detach().numpy(), dtype=torch.float32)
    dot = torch.mm(emb, emb.permute(1, 0))
    max_value = torch.max(dot)
    sim = torch.sigmoid(5 * dot / max_value)
    adj = torch.round(sim - 0.3)
    adj = sp.coo_matrix(adj.detach().numpy())
    return adj


def generate_syntax_adj_matrix(paper_ids, systax_embedings, paper_length):
    adj = [[0 for _ in range(paper_length)] for _ in range(paper_length)]

    for i in range(len(paper_ids)):
        embeding1 = systax_embedings[paper_ids[i]]

        for j in range(i + 1, len(paper_ids)):
            embeding2 = systax_embedings[paper_ids[j]]
            weight = cosine_similarity(embeding1, embeding2)
            adj[i][j] = adj[j][i] = weight

    adj = np.array(adj, dtype=np.float32).reshape((paper_length, paper_length))  # / maxValues
    adj[adj >= 0.8] = 1
    adj[adj < 0.8] = 0
    for i in range(len(paper_ids)):
        adj[i][i] = 1
    return adj

def get_config(config_path):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config




if __name__ == '__main__':
    print(np.argmax([0, 1, 2]))
