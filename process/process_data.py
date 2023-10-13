import os
import sys
# 解决linux下无法导入自己的包的问题
import copy
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
from src.new_models import NB_AREByS_N2VEmbGCN_SCL
from src.dataset import GCN_ContrastiveDataSetSR,DataLoader
from src.contrastive_loss import SupConLoss,GraphLoss
from src.utils import parseJson,saveJson,stat_add_edge_true_false_num,stat_rm_edge_true_false_num,evaluate,evaluate_fourMetrics,get_config
from src.util_training import setup_seed, mkdir, draw_acc_loss_curve
from src.clusters import paperClusterByDis
import torch
import torch.nn.functional as F
import datetime
import pandas as pd
import numpy as np
import codecs
import argparse
import wandb
import warnings
from pandas.core.common import SettingWithCopyWarning
# from pyinstrument import Profiler

###########################################################################
# warning 忽略
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
log = sys.stdout


def parse_configion(cfg_path="and/config/Aminer-18/SCL/cfg.yml"):
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--cfg_path', type=str, default=cfg_path,
                        help='path to the config file')
    parser.add_argument('--run_model', type=str, default="debug", choices=['run', 'debug'],
                        help='batch_size')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='mlp dropout')
    # temperature
    parser.add_argument('--temperature_content', type=float, default=0.07,
                        help='temperature for loss function')

    parser.add_argument('--temperature_structure', type=float, default=0.07,
                        help='temperature for loss function')

    parser.add_argument('--temperature_fusion', type=float, default=0.07,
                        help='temperature for loss function')


    parser.add_argument('--lossfac_content', type=float, default=0.5,
                        help='temperature for loss function')
    # parser.add_argument('--lossfac_structure', type=float, default=0.07,
    #                     help='temperature for loss function')
    parser.add_argument('--lossfac_fusion', type=float, default=1.0,
                        help='temperature for loss function')



    parser.add_argument('--hidden', type=int, default=150,
                        help='number of training epochs')
    parser.add_argument('--low_sim_threshold', type=float, default=0.20,
                        help='low sim_threshold for graph structure learning')
    parser.add_argument('--high_sim_threshold', type=float, default=0.80,
                        help='high sim_threshold for graph structure learning')
    parser.add_argument('--metric_type', type=str, default='cosine',
                        help='metric_type for graph structure learning')
    # configimization
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--gcnLayer', type=int, default=1,
                        help='gcnLayer')

    parser.add_argument('--seed', type=int, default=2021,
                        help='seed')
    parser.add_argument('--rel_freeze', type=int, default=1,
                        help='rel_freeze for rel embedding layer 1:True fix ;0:False free')
    parser.add_argument('--sem_freeze', type=int, default=1,
                        help='sem_freeze for sem embedding layer ')

    dymic_args = vars(parser.parse_args())
    static_args = get_config(dymic_args['cfg_path'])
    args = dict(dymic_args, **static_args)
    return args


config = parse_configion()

def fun1():
    # fun1: merge test and train raw pub data to all raw pub data

    all_raw_data_path = os.path.join(config['aminerData_path'],config['all_raw_pub_data'])
    test_raw_pub_data_path = os.path.join(config['aminerData_path'],config['test_raw_pub_data'])
    train_raw_pub_data_path = os.path.join(config['aminerData_path'],config['train_raw_pub_data'])
    valid_raw_pub_data_path = os.path.join(config['aminerData_path'],config['valid_raw_pub_data'])

    test_raw_pub_data = parseJson(test_raw_pub_data_path)
    train_raw_pub_data = parseJson(train_raw_pub_data_path)
    valid_raw_pub_data = parseJson(valid_raw_pub_data_path)
    all_raw_data_t = dict(train_raw_pub_data,**valid_raw_pub_data)
    all_raw_data = dict(all_raw_data_t,**test_raw_pub_data)

    saveJson(all_raw_data_path, all_raw_data)


def fun2():
    # fun2: get all_pid2name.json

    test_paper_label_df_path = os.path.join(config['aminerDataProcess_path'], config['test_df'])
    train_paper_label_df_path = os.path.join(config['aminerDataProcess_path'], config['train_df'])
    valid_paper_label_df_path = os.path.join(config['aminerDataProcess_path'], config['valid_df'])
    all_pid2name_path = os.path.join(config['aminerDataProcess_path'], config['all_pid2name'])

    train_paper_label_df = pd.read_csv(train_paper_label_df_path)
    valid_paper_label_df = pd.read_csv(valid_paper_label_df_path)
    test_paper_label_df = pd.read_csv(test_paper_label_df_path)

    all_paper_label_df = pd.concat([train_paper_label_df, valid_paper_label_df, test_paper_label_df], axis=0,
                                   ignore_index=True)
    all_pid2name = {}
    repeat = 1
    repeat_dif_name = 1
    for index,row in all_paper_label_df.iterrows():

        pid = row['paperid']
        name = row['name']
        if pid in all_pid2name.keys():
            print("repeat pid: {}".format(pid))
            repeat += 1
            if all_pid2name[pid] != name:
                repeat_dif_name+=1

        else:
            all_pid2name[pid] = name

    # saveJson(all_pid2name_path, all_pid2name)
    print(repeat,repeat_dif_name)



if __name__ == '__main__':
    fun1()
    fun2()
    # fun3()