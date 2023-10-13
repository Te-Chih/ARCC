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

# step1: parse config and add parse
config = parse_configion()
# build graph
config['build_graph'] = 'RULE'
# refin graph structure
config['refine_graph'] = 'AREByS'
# MODE
config['network'] = 'NB2MLP_N2VEMCLayer_AREGCN2MLP_FUSION2MLP_3temperature_3lossfac'
# wandb
config['wandb_run_name'] = config['refine_graph'] + '_' + config['build_graph'] + '_' + config['network']
# step2: 构造output file,定义要模型产生的输出的目录
curFileName = os.path.basename(__file__).split('.')[0]
save_base_folder = "{}/{}".format(config['save_path'], curFileName)
save_checkpoints = "{}/train_checkpoints".format(save_base_folder)
save_bestmodels = "{}/train_bestmodels".format(save_base_folder)
save_train_result = "{}/train_result".format(save_base_folder)
save_train_logs = "{}/train_logs".format(save_base_folder)
save_test_output = "{}/test_output".format(save_base_folder)
save_test_result = "{}/test_result".format(save_base_folder)

# step3: get file path
# embedding
paper_embeddings_path = os.path.join(config['aminerEmbedding_path'],config['all_semantic_emb_nb50'])
train_relation_embeddings_path = os.path.join(config['aminerEmbedding_path'],config['train_rel_emb_rule'])
test_relation_embeddings_path = os.path.join(config['aminerEmbedding_path'],config['test_rel_emb_rule'])

# adj
train_adj_matrix_path =  os.path.join(config['aminerDataProcess_path'],config['train_adj_rule'])
test_adj_matrix_path = os.path.join(config['aminerDataProcess_path'],config['test_adj_rule'])

# df
all_pid2name_path = os.path.join(config['aminerDataProcess_path'],config['all_pid2name'])
train_data_path = os.path.join(config['aminerDataProcess_path'],config['train_df'])
eval_data_path = os.path.join(config['aminerDataProcess_path'],config['valid_df'])#
test_data_path = os.path.join(config['aminerDataProcess_path'],config['test_df'])

# embedding layer
all_pid_list_path = os.path.join(config['aminerDataProcess_path'], config['all_pid_rule'])
all_pid_to_idx_path = os.path.join(config['aminerDataProcess_path'], config['all_pid_to_idx_rule'])
all_rel_emb_vector_path = os.path.join(config['aminerDataProcess_path'], config['all_rel_emb_vector_rule'])
all_sem_emb_vector_path = os.path.join(config['aminerDataProcess_path'], config['all_sem_emb_vector_rule'])

# 论文源数据
# paper_infos = parseJson(os.path.join(config['aminerData_path'],config['raw_data']))
test_raw_data_path = os.path.join(config['aminerData_path'],config['test_raw_data'])
###########################################################################
ACC_SIM = config['acc_sim']
def train():
    epochs = config['epochs']
    batch_size = config['batch_size']
    hidden = config['hidden']
    dropout = config['dropout']
    lr = config['learning_rate']
    temperature_content = config['temperature_content']
    temperature_structure = config['temperature_structure']
    temperature_fusion = config['temperature_fusion']
    lossfac_content = config['lossfac_content']
    lossfac_structure = 1.0 - lossfac_content
    lossfac_fusion = config['lossfac_fusion']
    # temperature = config['temperature']
    run_model = config['run_model']
    gcnLayer = config['gcnLayer']
    seed = config['seed']
    low_sim_threshold = config['low_sim_threshold']
    high_sim_threshold = config['high_sim_threshold']
    metric_type = config['metric_type']
    rel_freeze = bool(config['rel_freeze'])
    sem_freeze = bool(config['sem_freeze'])


    save_file_name = 'model{}_sem{}_rel{}_dp{}_hid{}_ep{}_bs{}_lr{}_tepc{}_tepr{}_tepf{}_lfc{}_lfs{}_lff{}_gcnL{}_seed{}_metric-{}_low{}_high{}'.format(
        run_model,sem_freeze,rel_freeze,dropout, hidden, epochs, batch_size, lr, temperature_content,temperature_structure,temperature_fusion,lossfac_content,lossfac_structure,lossfac_fusion,gcnLayer,seed,metric_type,low_sim_threshold,high_sim_threshold)
    logpath = "{}/{}.txt".format(save_train_logs, save_file_name)

    global log
    if run_model == "debug":
        log = sys.stdout
    else:
        log = open(logpath, "w", encoding="utf-8")




    # adj
    train_adj_matrix = parseJson(train_adj_matrix_path)
    # pid2name
    all_pid2name = parseJson(all_pid2name_path)

    # df
    train_data_df = pd.read_csv(train_data_path)
    eval_data_df = pd.read_csv(eval_data_path)

    #name list
    train_name_list = train_data_df['name'].unique().tolist()
    eval_name_list = eval_data_df['name'].unique().tolist()

    # embedding layer data
    # all_pid_list = parseJson(all_pid_list_path)
    all_pid_to_idx = parseJson(all_pid_to_idx_path)
    all_rel_emb_vector_list = parseJson(all_rel_emb_vector_path)
    all_rel_emb_vector = torch.tensor(all_rel_emb_vector_list).cuda()
    all_sem_emb_vector_list = parseJson(all_sem_emb_vector_path)
    all_sem_emb_vector = torch.tensor(all_sem_emb_vector_list).cuda()



    # dataset dataloader
    train_data = GCN_ContrastiveDataSetSR(train_name_list,train_data_df)
    training_params = {"batch_size": batch_size, "shuffle": True, "drop_last": False}
    training_generator = DataLoader(train_data, **training_params)

    eval_data = GCN_ContrastiveDataSetSR(eval_name_list,eval_data_df)
    eval_params = {"batch_size": batch_size, "shuffle": False, "drop_last": False}
    eval_generator = DataLoader(eval_data, **eval_params)

    #
    num_iter_per_epoch = len(training_generator)

    # model structure
    model = NB_AREByS_N2VEmbGCN_SCL(sem_freeze=sem_freeze,rel_freeze=rel_freeze,sem_emb_vector=all_sem_emb_vector,rel_emb_vector=all_rel_emb_vector,
                                    hidden=hidden, dropout=dropout,gcn_layer= gcnLayer,low_sim_threshold=low_sim_threshold,
                                    high_sim_threshold = high_sim_threshold,metric_type=metric_type).cuda()
    # loss function
    criterion_content = SupConLoss(temperature=temperature_content)
    criterion_structure = SupConLoss(temperature=temperature_structure)
    criterion_fusion = SupConLoss(temperature=temperature_fusion)
    graploss = GraphLoss(smoothness_ratio=0.2, degree_ratio=0, sparsity_ratio=0)
    # optimizer selector
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # var data
    train_step=0
    valid_step=0
    best_acc = 0
    best_loss = 100
    best_f1 = 0
    train_loss = []
    train_accs = []
    test_loss = []
    test_accs = []
    test_pres = []
    test_recs = []
    test_f1s = []
    best_modelPath = ""
    res_file_path = ""
    train_pres_perepoch=[]
    train_recs_perepoch=[]
    train_f1s_perepoch=[]
    train_add_edge_num_epoch = {}
    train_true_edge_num_epoch = {}
    train_false_edge_num_epoch = {}
    eval_add_edge_num_epoch = {}
    eval_true_edge_num_epoch = {}
    eval_false_edge_num_epoch = {}
    train_rm_edge_num_epoch = {}
    train_rm_true_edge_num_epoch = {}
    train_rm_false_edge_num_epoch = {}
    eval_rm_edge_num_epoch = {}
    eval_rm_true_edge_num_epoch = {}
    eval_rm_false_edge_num_epoch = {}



    for ep in range(epochs):
        if run_model == "debug":
            if ep > 2:
                break
        begin1 = datetime.datetime.now()

        print("##########{}/{}:Train Model#############".format(ep, epochs),file=log, flush=True)
        print("##########{}/{}:Train Model#############".format(ep, epochs))
        train_pres_perbs = []
        train_recs_perbs = []
        train_f1s_perbs = []
        train_loss_tmp = []
        train_acc_tmp = []
        model.train()
        begin2 = datetime.datetime.now()
        for iter, sample_list in enumerate(training_generator):
            train_step += 1
            name = all_pid2name[sample_list[0][0][0]]
            # 是否debug 模式
            if run_model == "debug":
                if iter > 2:
                    break

            label_list = []
            pid_index_list = []
            for paper_id,la in zip(sample_list[0],sample_list[1]):
                pid_index_list.append(all_pid_to_idx[paper_id[0]])
                label_list.append(la.item())
            pid_index_tensor = torch.tensor(pid_index_list).cuda()
            label = torch.tensor(label_list).cuda()
            adj_matrix_tensor = torch.tensor(train_adj_matrix[name],requires_grad=True).cuda()

            # 梯度清零
            optimizer.zero_grad()
            # 执行模型，得到批预测结果
            s_emb, r_emb, prediction, refine_adj_matrix_tensor = model(pid_index_tensor=pid_index_tensor,adj_matrix_tensor=adj_matrix_tensor)

            # 根据pre与label值的距离，计算loss，loss是标量
            # loss 是数值，是通过loss function计算得到的数值，含义是预测数值与真实数值的差距；
            loss1 = criterion_content(s_emb, label)
            loss2 = criterion_structure(r_emb, label)
            loss3 = criterion_fusion(prediction, label)
            loss = lossfac_content*loss1 + lossfac_structure*loss2 + lossfac_fusion*loss3
            # 误差反向传播，计算梯度
            loss.backward()
            # 根据梯度，更新参数
            optimizer.step()

            add_edge_num, true_edge_num, false_edge_num = stat_add_edge_true_false_num(adj_matrix_tensor, refine_adj_matrix_tensor, label)
            rm_edge_num, rm_true_edge_num, rm_false_edge_num = stat_rm_edge_true_false_num(adj_matrix_tensor, refine_adj_matrix_tensor, label)
            train_add_edge_num_epoch.setdefault(name, []).append(add_edge_num)
            train_true_edge_num_epoch.setdefault(name, []).append(true_edge_num)
            train_false_edge_num_epoch.setdefault(name, []).append(false_edge_num)
            train_rm_edge_num_epoch.setdefault(name, []).append(rm_edge_num)
            train_rm_true_edge_num_epoch.setdefault(name, []).append(rm_true_edge_num)
            train_rm_false_edge_num_epoch.setdefault(name, []).append(rm_false_edge_num)
            # 得到accuracy：预测正确的样本个数与总样本数的比值，需要对比预测的类别标签，与真实值得类别标签
            ########## computer_ACC####################

            sim_matrix = F.cosine_similarity(prediction.unsqueeze(1), prediction.unsqueeze(0), dim=2).detach()
            pred_matrix = torch.where(sim_matrix > ACC_SIM, 1, 0).detach()
            label_matrix = torch.where(label.unsqueeze(1) == label.unsqueeze(1).T, 1, 0).detach()
            acc_t = torch.sum(torch.where(label_matrix == pred_matrix, 1, 0)).item() / (
                    label_matrix.shape[0] * label_matrix.shape[1])
            # train_acc_iter = np.mean(acc_t)
            train_acc_tmp.append(acc_t)
            train_loss_tmp.append(loss.item())
            ###### computer F1 #######
            dis = 1 - sim_matrix
            cluster_num = len(set(label_list))
            papers = list(range(len(label_list)))
            paper_name_dict = {}
            for paperid in papers:
                label_value=int(label[paperid])
                if label_value not in paper_name_dict.keys():
                    paper_name_dict[label_value] = []
                    paper_name_dict[label_value].append(paperid)
                else:
                    paper_name_dict[label_value].append(paperid)
            result = paperClusterByDis(dis.cpu(), papers, cluster_num, method='AG')
            # 评估指标
            precision, recall, f1 = evaluate(result, paper_name_dict)
            # tp, fp, fn, tn = evaluate_fourMetrics(result, name_papers[name])
            train_pres_perbs.append(precision)
            train_recs_perbs.append(recall)
            train_f1s_perbs.append(f1)

            wandb.log({"train/step_acc": acc_t, "train/step_loss": loss.item(), "train/step_pre": precision,
                       "train/step_recall": recall, "train/step_f1": f1, "train/step": train_step})

            # 打印指标日志
            end2 = datetime.datetime.now()
            if iter % 100 == 0:
                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {:.4f}, Accuracy: {},Usingtime:{}".format(ep + 1,epochs,iter + 1,num_iter_per_epoch,optimizer.param_groups[0]['lr'],loss,acc_t,end2 - begin2),file=log, flush=True)
                print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {:.4f}, Accuracy: {},Usingtime:{}".format(ep + 1,epochs,iter + 1,num_iter_per_epoch,optimizer.param_groups[0]['lr'],loss,acc_t,end2 - begin2))

            begin2 = datetime.datetime.now()

        train_loss.append(np.mean(train_loss_tmp))
        train_accs.append(np.mean(train_acc_tmp))
        train_pres_perepoch.append(np.mean(train_pres_perbs))
        train_recs_perepoch.append(np.mean(train_recs_perbs))
        train_f1s_perepoch.append(np.mean(train_f1s_perbs))

        wandb.log({"train/epoch_avg_acc": np.mean(train_acc_tmp), "train/epoch_avg_loss": np.mean(train_loss_tmp),
                   "train/epoch_avg_pre": np.mean(train_pres_perbs),
                   "train/epoch_avg_recall": np.mean(train_recs_perbs), "train/epoch_avg_f1": np.mean(train_f1s_perbs),
                   "epochs": ep})


        print("##########{}/{}: Eval Model#############".format(ep,epochs))
        print("##########{}/{}: Eval Model#############".format(ep, epochs),file=log, flush=True)
        #验证该epoch 的性能
        # model 处于 eval状态
        model.eval()
        accs = []
        losses = []
        counter = 0
        # 不记录梯度
        with torch.no_grad():
            test_pres_perepoch = []
            test_recs_perepoch = []
            test_f1s_perepoch = []
            for iter, sample_list in enumerate(eval_generator):
                valid_step+=1
                name = all_pid2name[sample_list[0][0][0]]

                if run_model == "debug":
                    if iter > 2:
                        break

                label_list = []
                pid_index_list = []
                for paper_id, la in zip(sample_list[0], sample_list[1]):
                    pid_index_list.append(all_pid_to_idx[paper_id[0]])
                    label_list.append(la.item())
                pid_index_tensor = torch.tensor(pid_index_list).cuda()
                label = torch.tensor(label_list).cuda()
                adj_matrix_tensor = torch.tensor(train_adj_matrix[name]).cuda()



                s_emb, r_emb, prediction,refine_adj_matrix_tensor = model(pid_index_tensor=pid_index_tensor, adj_matrix_tensor=adj_matrix_tensor)

                # 根据pre与label值的距离，计算loss，loss是标量
                # loss 是数值，是通过loss function计算得到的数值，含义是预测数值与真实数值的差距；
                loss1 = criterion_content(s_emb, label)
                loss2 = criterion_structure(r_emb, label)
                loss3 = criterion_fusion(prediction, label)
                loss = lossfac_content * loss1 + lossfac_structure * loss2 + lossfac_fusion * loss3

                add_edge_num, true_edge_num, false_edge_num = stat_add_edge_true_false_num(adj_matrix_tensor,
                                                                                           refine_adj_matrix_tensor,
                                                                                           label)
                rm_edge_num, rm_true_edge_num, rm_false_edge_num = stat_rm_edge_true_false_num(adj_matrix_tensor,
                                                                                               refine_adj_matrix_tensor,
                                                                                               label)
                eval_add_edge_num_epoch.setdefault(name, []).append(add_edge_num)
                eval_true_edge_num_epoch.setdefault(name, []).append(true_edge_num)
                eval_false_edge_num_epoch.setdefault(name, []).append(false_edge_num)
                eval_rm_edge_num_epoch.setdefault(name, []).append(rm_edge_num)
                eval_rm_true_edge_num_epoch.setdefault(name, []).append(rm_true_edge_num)
                eval_rm_false_edge_num_epoch.setdefault(name, []).append(rm_false_edge_num)


                # 得到accuracy：预测正确的样本个数与总样本数的比值，需要对比预测的类别标签，与真实值得类别标签
                ########## computer_ACC####################
                sim_matrix = F.cosine_similarity(prediction.unsqueeze(1), prediction.unsqueeze(0), dim=2).detach()
                pred_matrix = torch.where(sim_matrix > ACC_SIM, 1, 0).detach()
                label_matrix = torch.where(label.unsqueeze(1) == label.unsqueeze(1).T, 1, 0).detach()
                acc_t = torch.sum(torch.where(label_matrix == pred_matrix, 1, 0)).item() / (
                        label_matrix.shape[0] * label_matrix.shape[1])
                accs.append(acc_t)
                losses.append(loss.item())
                counter += 1
                ###### computer F1 #######
                dis = 1 - sim_matrix
                cluster_num = len(set(label_list))
                papers = list(range(len(label_list)))
                paper_name_dict = {}
                for paperid in papers:
                    label_value = int(label[paperid])
                    if label_value not in paper_name_dict.keys():
                        paper_name_dict[label_value] = []
                        paper_name_dict[label_value].append(paperid)
                    else:
                        paper_name_dict[label_value].append(paperid)
                result = paperClusterByDis(dis.cpu(), papers, cluster_num, method='AG')
                # 评估指标
                precision, recall, f1 = evaluate(result, paper_name_dict)
                test_pres_perepoch.append(precision)
                test_recs_perepoch.append(recall)
                test_f1s_perepoch.append(f1)

                wandb.log({"valid/step_acc": acc_t, "valid/step_loss": loss.item(), "valid/step_pre": precision,
                           "valid/step_recall": recall, "valid/step_f1": f1, "valid/step": valid_step})
            # 计算平均loss, 总loss / 总数据个数
            te_loss = sum(losses) / counter
            # 记录该批次的evel数据集的loss
            test_loss.append(te_loss)
            te_acc = sum(accs) / counter
            test_accs.append(te_acc)
            test_pres.append(np.mean(test_pres_perepoch))
            test_recs.append(np.mean(test_recs_perepoch))
            test_f1s.append(np.mean(test_f1s_perepoch))
            te_f1= np.mean(test_f1s_perepoch)
            if te_f1 > best_f1:
                best_f1 = te_f1
                # best_loss = te_loss
                best_model_epoch = ep
                #todo 添加参数
                best_modelPath = "{}/{}.pkt".format(save_bestmodels, save_file_name)
                torch.save(model, best_modelPath)
            wandb.log({"valid/epoch_avg_acc": te_acc, "valid/epoch_avg_loss": te_loss,
                       "valid/epoch_avg_pre": np.mean(test_pres_perepoch),
                       "valid/epoch_avg_recall": np.mean(test_recs_perepoch),
                       "valid/epoch_avg_f1": np.mean(test_f1s_perepoch), "epochs": ep,"valid/epoch_valid_best_f1":best_f1})
            # 打印eval的指标日志
            print("Eval ==> Epoch: {}/{}, EvalLoss: {:.4f}, EvalAccuracy: {}".format(ep, epochs, te_loss, te_acc),file=log, flush=True)
            print("Eval ==> Epoch: {}/{}, EvalLoss: {:.4f}, EvalAccuracy: {}".format(ep, epochs, te_loss, te_acc))
        # 最好的epoch数据记录下来
        #todo 添加参数
        result = {
            'model_name': curFileName,
            'run_model':run_model,
            'best_model_path': best_modelPath,
            'best_model_epoch': best_model_epoch,

            'log_path' : logpath,
            'parameters': {
                'hidden': hidden,
                'epoch': epochs,
                'batch_size': batch_size,
                'lr': lr,
                'dropout': dropout,
                'temperature_content': temperature_content,
                'temperature_structure': temperature_structure,
                'temperature_fusion': temperature_fusion,
                'lossfac_content': lossfac_content,
                'lossfac_structure': lossfac_structure,
                'lossfac_fusion': lossfac_fusion,
                'gcnLayer':gcnLayer,
                'seed':seed,
                'metric_type': metric_type,
                'low_sim_threshold':low_sim_threshold,
                'high_sim_threshold':high_sim_threshold,
                'rel_freeze':rel_freeze,
                'sem_freeze':sem_freeze
            },
            'best_eval_acc': best_acc,
            'best_eval_loss': best_loss,
            'best_eval_f1': best_f1,
            'train_accs': train_accs,
            'train_loss': train_loss,
            'train_pres': train_pres_perepoch,
            'train_recs': train_recs_perepoch,
            'train_f1s': train_f1s_perepoch,
            'eval_accs': test_accs,
            'eval_loss': test_loss,
            'eval_pres': test_pres,
            'eval_recs': test_recs,
            'eval_f1s': test_f1s,
            'train_add_edge_num_epoch':train_add_edge_num_epoch,
            'train_true_edge_num_epoch':train_true_edge_num_epoch,
            'train_false_edge_num_epoch':train_false_edge_num_epoch,
            'eval_add_edge_num_epoch':eval_add_edge_num_epoch,
            'eval_true_edge_num_epoch':eval_true_edge_num_epoch,
            'eval_false_edge_num_epoch':eval_false_edge_num_epoch,
            'train_rm_edge_num_epoch':train_rm_edge_num_epoch,
            'train_rm_true_edge_num_epoch':train_rm_true_edge_num_epoch,
            'train_rm_false_edge_num_epoch':train_rm_false_edge_num_epoch,
            'eval_rm_edge_num_epoch':eval_rm_edge_num_epoch,
            'eval_rm_true_edge_num_epoch':eval_rm_true_edge_num_epoch,
            'eval_rm_false_edge_num_epoch':eval_rm_false_edge_num_epoch
        }
        res_file_path = "{}/{}.json".format(save_train_result,save_file_name)
        saveJson(res_file_path, result)
        end1 = datetime.datetime.now()
        print("one epoch using time:", end1 - begin1,file=log, flush=True)
        print("one epoch using time:", end1 - begin1)
    print("################ TrainFunction Finish! ##########################",file=log, flush=True)
    print("################ TrainFunction Finish! ##########################")

    return res_file_path

def model_test(res_file_path):
    train_res_file=parseJson(res_file_path)
    best_modelPath = train_res_file["best_model_path"]
    global log
    model_name = best_modelPath.split("/")[-1].split(".pkt")[0]
    print("################# Test Start ###################", file=log, flush=True)
    print("################# Test Start ###################")
    # test数据集
    name_papers = parseJson(test_raw_data_path)
    test_adj_matrix = parseJson(test_adj_matrix_path)

    all_pid_to_idx = parseJson(all_pid_to_idx_path)

    model = torch.load(best_modelPath).cuda()
    model.eval()

    result_file_word2vec_local1 = "{}/{}.xls".format(save_test_output,model_name)
    file_word2vec_local1 = codecs.open(result_file_word2vec_local1, "w")
    file_word2vec_local1.write("index\tname\tprecision\trecall\tf1\n")

    sigmoid_score1 = [0, 0, 0,0,0,0,0]
    cnt = 0
    train_acc_tmp =[]
    test_add_edge_num_epoch = {}
    test_true_edge_num_epoch = {}
    test_false_edge_num_epoch = {}
    test_rm_edge_num_epoch = {}
    test_rm_true_edge_num_epoch = {}
    test_rm_false_edge_num_epoch = {}
    # 按name迭代预测
    for index, name in enumerate(name_papers.keys()):
        # try:
            cnt += 1
            papers = []
            # 获得该name的所有论文
            label_list = []
            label_counter = 1

            for talentid in name_papers[name]:
                papers.extend(name_papers[name][talentid])
                label_list.extend([label_counter] * len(name_papers[name][talentid]))
                label_counter += 1

            pid_index_list = []
            for paper_id in papers:
                pid_index_list.append(all_pid_to_idx[paper_id])

            pid_index_tensor = torch.tensor(pid_index_list).cuda()
            adj_matrix_tensor = torch.tensor(test_adj_matrix[name], requires_grad=True).cuda()
            label = torch.tensor(label_list).cuda()

            paper_num = len(papers)
            print(index, name, paper_num,file= log)
            print(index, name, paper_num)



            s_emb, r_emb, prediction,refine_adj_matrix_tensor = model(pid_index_tensor=pid_index_tensor,adj_matrix_tensor=adj_matrix_tensor)


            pred = prediction.cpu().detach()


            ##############analysis add edge num######################
            add_edge_num, true_edge_num, false_edge_num = stat_add_edge_true_false_num(adj_matrix_tensor,
                                                                                       refine_adj_matrix_tensor,
                                                                                       label.cuda())

            test_add_edge_num_epoch.setdefault(name, []).append(add_edge_num)
            test_true_edge_num_epoch.setdefault(name, []).append(true_edge_num)
            test_false_edge_num_epoch.setdefault(name, []).append(false_edge_num)

            rm_edge_num, rm_true_edge_num, rm_false_edge_num = stat_rm_edge_true_false_num(adj_matrix_tensor,
                                                                                       refine_adj_matrix_tensor,
                                                                                       label.cuda())

            test_rm_edge_num_epoch.setdefault(name, []).append(rm_edge_num)
            test_rm_true_edge_num_epoch.setdefault(name, []).append(rm_true_edge_num)
            test_rm_false_edge_num_epoch.setdefault(name, []).append(rm_false_edge_num)

            #####################################

            # acc_t = SupCon_Acc_compute(paper_num, pred, label)
            sim_matrix = F.cosine_similarity(pred.unsqueeze(1), pred.unsqueeze(0), dim=2)
            pred_matrix = torch.where(sim_matrix > ACC_SIM, 1, 0)
            label_matrix = torch.where(label.unsqueeze(1) == label.unsqueeze(1).T, 1, 0).detach()
            label_matrix = label_matrix.cpu().detach()
            pred_matrix = pred_matrix.cpu().detach()
            # train_acc_iter = np.mean(acc_t)
            train_metrics_t = torch.sum(torch.where(label_matrix == pred_matrix, 1, 0)).item() / (
                    label_matrix.shape[0] * label_matrix.shape[1])
            train_acc_tmp.append(train_metrics_t)


            pred_sim = sim_matrix
            dis = 1 - pred_sim
            # papers 所有的论文，name_papers[name] 所有的talentid数量，即为聚类数量
            # 聚类算法，
            result = paperClusterByDis(dis, papers, len(name_papers[name]), method='AG')
            # 评估指标
            precision, recall, f1 = evaluate(result, name_papers[name])
            tp, fp, fn, tn = evaluate_fourMetrics(result, name_papers[name])
            sigmoid_score1[0] += precision
            sigmoid_score1[1] += recall
            sigmoid_score1[2] += f1
            sigmoid_score1[3] += tp
            sigmoid_score1[4] += fp
            sigmoid_score1[5] += fn
            sigmoid_score1[6] += tn
            wandb.log({"test/step_acc": train_metrics_t, "test/step_pre": precision,
                       "test/step_recall": recall, "test/step_f1": f1, "test/index": index})
            file_word2vec_local1.write("%s\t %s\t %4.2f%%\t %4.2f%%\t %4.2f%%\n" % (index, name, precision * 100, recall * 100, f1 * 100))
            # for()
            print("pred距离聚类结果：", precision, recall, f1, file=log, flush=True)
            print("pred距离聚类结果：", precision, recall, f1)

        # except Exception as e:
        #     print(e)


    sigmoid_score1[0] /= cnt
    sigmoid_score1[1] /= cnt
    sigmoid_score1[2] /= cnt
    sigmoid_score1[3] /= cnt
    sigmoid_score1[4] /= cnt
    sigmoid_score1[5] /= cnt
    sigmoid_score1[6] /= cnt
    wandb.log({"test/avg_acc": np.mean(train_acc_tmp), "test/avg_pre": sigmoid_score1[0],
               "test/avg_recall": sigmoid_score1[1], "test/avg_f1": sigmoid_score1[2]})
    file_word2vec_local1.write(
        "0\t average\t %4.2f%%\t %4.2f%%\t %4.2f%%\n" % (
            sigmoid_score1[0] * 100, sigmoid_score1[1] * 100, sigmoid_score1[2] * 100))
    train_res_file["test_add_edge_num_epoch"] = test_add_edge_num_epoch
    train_res_file["test_true_edge_num_epoch"] = test_true_edge_num_epoch
    train_res_file["test_false_edge_num_epoch"] = test_false_edge_num_epoch
    train_res_file["test_rm_edge_num_epoch"] = test_rm_edge_num_epoch
    train_res_file["test_rm_true_edge_num_epoch"] = test_rm_true_edge_num_epoch
    train_res_file["test_rm_false_edge_num_epoch"] = test_rm_false_edge_num_epoch
    train_res_file["test_metrics"] = "0\t average\t %4.2f%%\t %4.2f%%\t %4.2f%%\n" % (
            sigmoid_score1[0] * 100, sigmoid_score1[1] * 100, sigmoid_score1[2] * 100)
    train_res_file["test_metrics-v2"] = "0\t average\t TP: %4.2f%%\t FP: %4.2f%%\t FN:%4.2f%%\t TN:%4.2f%%\n" % (
        sigmoid_score1[3] * 100, sigmoid_score1[4] * 100, sigmoid_score1[5] * 100, sigmoid_score1[6] * 100)
    train_res_file["test_acc"] = np.mean(train_acc_tmp)
    test_res_file_path = "{}/{}.json".format(
        save_test_result, model_name)
    saveJson(test_res_file_path, train_res_file)
    print("################ TestFunction Finish! ##########################", file=log, flush=True)
    print("################ TestFunction Finish! ##########################")
    return test_res_file_path


if __name__ == '__main__':
    # step1: wandb init
    wandb.init(project=config['wandb_project_name'],
               name=config['wandb_run_name'] + "_" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               config=config)

    # step2: setup_seed
    setup_seed(config['seed'])

    # step3: mk output file
    mkdir(save_base_folder)
    mkdir(save_checkpoints)
    mkdir(save_bestmodels)
    mkdir(save_train_result)
    mkdir(save_train_logs)
    mkdir(save_test_output)
    mkdir(save_test_result)

    # step4: train and test and draw_pic:
    res_file_path = train()
    test_res_file_path = model_test(res_file_path)
    draw_acc_loss_curve(test_res_file_path, save_test_result)
    wandb.finish()

"""
CUDA_VISIBLE_DEVICE=1 nohup python train_contrastive_loss_sr.py --epochs 50 \
--batch_size 32 \
--hidden 100 \
--dropout 0.5 \
--temperature 0.07\
--learning_rate 0.001 > log-6.txt 2>&1 &
"""
