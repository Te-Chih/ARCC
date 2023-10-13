import os
import sys
# 解决linux下无法导入自己的包的问题
import copy
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
import argparse
from tqdm import tqdm
from src.utils import parseJson,saveJson,get_config
def parse_configion(cfg_path="and/config/WhoIsWho/SCL/cfg_50.yml"):
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--cfg_path', type=str, default=cfg_path,
                        help='path to the config file')
    dymic_args = vars(parser.parse_args())
    static_args = get_config(dymic_args['cfg_path'])
    args = dict(dymic_args, **static_args)
    return args

config = parse_configion()


#50  3co
paper_embeddings_path = os.path.join(config['aminerEmbedding_path'],config['all_semantic_emb_nb50'])
train_relation_embeddings_path = os.path.join(config['aminerEmbedding_path'],config['train_rel_emb_3co'])
test_relation_embeddings_path = os.path.join(config['aminerEmbedding_path'],config['test_rel_emb_3co'])
valid_relation_embeddings_path = os.path.join(config['aminerEmbedding_path'],config['valid_rel_emb_3co'])

all_pid_list_path = os.path.join(config['aminerDataProcess_path'], config['all_pid_3co'])
all_pid_to_idx_path = os.path.join(config['aminerDataProcess_path'], config['all_pid_to_idx_3co'])
all_rel_emb_vector_path = os.path.join(config['aminerDataProcess_path'], config['all_rel_emb_vector_3co'])
all_b50_sem_emb_vector_path = os.path.join(config['aminerDataProcess_path'], config['all_b50_sem_emb_vector_3co'])


def produce_EmbeddingLayer_date():
    all_sem_emb_list = parseJson(paper_embeddings_path)
    train_rel_emb_list = parseJson(train_relation_embeddings_path)
    test_rel_emb_list = parseJson(test_relation_embeddings_path)
    valid_rel_emb_list = parseJson(valid_relation_embeddings_path)
    all_rel_emb_list_t = dict(train_rel_emb_list, **valid_rel_emb_list)
    all_rel_emb_list = dict(all_rel_emb_list_t, **test_rel_emb_list)

    # pid_list = [pid.split('-')[0] for pid in all_rel_emb_list.keys()]
    rel_pid_list = [pid for pid in all_rel_emb_list.keys()]
    # sem_pid_list = [pid for pid in all_sem_emb_list.keys()]
    print(len(rel_pid_list),len(all_sem_emb_list))
    # rel_pid_to_idx = {pid:index for index,pid in enumerate(all_rel_pid_list)}
    pid_to_idx={}
    rel_emb_vector = [0]*len(rel_pid_list)
    sem_emb_vector = [0]*len(rel_pid_list)
    pid_list = [0] * len(rel_pid_list)
    for index, r_pid in enumerate(tqdm(rel_pid_list)):
        # pid = r_pid.split('-')[0]
        pid_list[index] = r_pid
        pid_to_idx[r_pid] = index
        rel_emb_vector[index] = all_rel_emb_list[r_pid]
        sem_emb_vector[index]= all_sem_emb_list[r_pid]



    saveJson(all_pid_list_path,pid_list)
    saveJson(all_pid_to_idx_path,pid_to_idx)
    saveJson(all_rel_emb_vector_path,rel_emb_vector)
    saveJson(all_b50_sem_emb_vector_path,sem_emb_vector)
    print("finish...")



def produce_EmbeddingLayer_date_50():
    all_sem_emb_list = parseJson(paper_embeddings_path)
    train_rel_emb_list = parseJson(train_relation_embeddings_path)
    test_rel_emb_list = parseJson(test_relation_embeddings_path)
    valid_rel_emb_list = parseJson(valid_relation_embeddings_path)
    all_rel_emb_list_t = dict(train_rel_emb_list, **valid_rel_emb_list)
    all_rel_emb_list = dict(all_rel_emb_list_t, **test_rel_emb_list)

    new_all_rel_emb_list = {}
    new_all_sem_emb_list = {}
    for name,pid_list in all_rel_emb_list.items():
        for pid,emb in pid_list.items():
            new_all_rel_emb_list[pid] = emb

    for pid, emb in all_sem_emb_list.items():
        # for pid, emb in pid_list.items():
        new_all_sem_emb_list[pid] = emb

    # pid_list = [pid.split('-')[0] for pid in all_rel_emb_list.keys()]
    rel_pid_list = [pid  for pid in new_all_rel_emb_list.keys()]

    pid_to_idx={}
    rel_emb_vector = [0]*len(rel_pid_list)
    sem_emb_vector = [0]*len(rel_pid_list)
    pid_list = [0] * len(rel_pid_list)

    for index, r_pid in enumerate(tqdm(rel_pid_list)):
        # pid = r_pid.split('-')[0]
        pid_list[index] = r_pid
        pid_to_idx[r_pid] = index
        rel_emb_vector[index] = new_all_rel_emb_list[r_pid]
        sem_emb_vector[index]= new_all_sem_emb_list[r_pid]



    saveJson(all_pid_list_path,pid_list)
    saveJson(all_pid_to_idx_path,pid_to_idx)
    saveJson(all_rel_emb_vector_path,rel_emb_vector)
    saveJson(all_b50_sem_emb_vector_path,sem_emb_vector)
    print("finish...")


def produce_EmbeddingLayer_date_3co():
    all_sem_emb_list = parseJson(paper_embeddings_path)
    train_rel_emb_list = parseJson(train_relation_embeddings_path)
    test_rel_emb_list = parseJson(test_relation_embeddings_path)
    valid_rel_emb_list = parseJson(valid_relation_embeddings_path)
    all_rel_emb_list_t = dict(train_rel_emb_list, **valid_rel_emb_list)
    all_rel_emb_list = dict(all_rel_emb_list_t, **test_rel_emb_list)

    # new_all_rel_emb_list = {}
    new_all_sem_emb_list = {}
    # for pid,emb in all_rel_emb_list:
    #     # for pid,emb in pid_list.items():
    #     new_all_rel_emb_list[pid] = emb

    for pid, emb in all_sem_emb_list.items():
        # for pid, emb in pid_list.items():
        new_all_sem_emb_list[pid] = emb

    # pid_list = [pid.split('-')[0] for pid in all_rel_emb_list.keys()]
    rel_pid_list = [pid  for pid in all_rel_emb_list.keys()]
    # rel_pid_list = [pid  for name_dict in all_rel_emb_list.keys() for pid in name_dict.keys()]
    # sem_pid_list = [pid for pid in all_sem_emb_list.keys()]
    # print(len(rel_pid_list),len(all_sem_emb_list))
    # rel_pid_to_idx = {pid:index for index,pid in enumerate(all_rel_pid_list)}
    pid_to_idx={}
    rel_emb_vector = [0]*len(rel_pid_list)
    sem_emb_vector = [0]*len(rel_pid_list)
    pid_list = [0] * len(rel_pid_list)

    for index, r_pid in enumerate(tqdm(rel_pid_list)):
        # pid = r_pid.split('-')[0]
        pid_list[index] = r_pid
        pid_to_idx[r_pid] = index
        rel_emb_vector[index] = all_rel_emb_list[r_pid]
        sem_emb_vector[index]= new_all_sem_emb_list[r_pid]



    saveJson(all_pid_list_path,pid_list)
    saveJson(all_pid_to_idx_path,pid_to_idx)
    saveJson(all_rel_emb_vector_path,rel_emb_vector)
    saveJson(all_b50_sem_emb_vector_path,sem_emb_vector)
    print("finish...")


if __name__ == '__main__':
    produce_EmbeddingLayer_date_3co()
