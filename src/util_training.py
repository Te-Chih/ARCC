import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib
from src.utils import parseJson
import scipy.sparse as sp
import numpy as np
import torch
import torch.nn.functional as F


def produce_prototype(features, labels):
    # 将标签转换为独热编码张量
    unique_labels, label_indices = torch.unique(labels, sorted=True, return_inverse=True)
    # unique_labels = unique_labels.cuda()
    # label_indices = label_indices.cuda()
    eye_num = torch.tensor(unique_labels.size(0)).cuda()
    one_hot_labels = torch.eye(eye_num).cuda()[label_indices.cuda()]

    # 计算每个类别的样本数
    sample_counts = one_hot_labels.sum(dim=0)

    # 计算每个类别的平均特征向量
    avg_features = torch.matmul(one_hot_labels.T, features) / sample_counts.unsqueeze(1)

    # 将每个平均特征对应的标签存储在列表中
    # avg_labels = unique_labels.tolist()
    return avg_features, unique_labels

def produce_prototype_random50(features, labels):
    # 统计不同的标签类别
    unique_labels = torch.unique(labels)
    print(unique_labels,unique_labels.shape)
    # 对于每个标签，取出对应的特征并求平均
    avg_features = []
    selection_ratio = 0.5
    for label in unique_labels:
        mask = (labels == label)  # 找出所有标签为label的样本的掩码
        label_features = features[mask]  # 从特征矩阵中取出所有标签为label的特征
        num_examples = label_features.shape[0]
        num_selected_examples = max(int(selection_ratio * num_examples), 1)
        # 随机选择示例的索引
        random_indices = random.sample(range(num_examples), num_selected_examples)
        # 从特征矩阵中获取随机选择的示例
        selected_examples = label_features[random_indices]
        avg_feature = torch.mean(selected_examples, dim=0)  # 对这些特征求平均
        avg_features.append(avg_feature)

    avg_features = torch.stack(avg_features)
    # avg_labels = unique_labels

    return avg_features, unique_labels


def produce_hard_negative_05(s_emb,r_emb,all_emb,labels):
    s_sim_matrix = F.cosine_similarity(s_emb.unsqueeze(1), s_emb.unsqueeze(0), dim=2)
    s_pred = torch.where(s_sim_matrix > 0.5, 1.0, 0.0)
    r_sim_matrix = F.cosine_similarity(r_emb.unsqueeze(1), r_emb.unsqueeze(0), dim=2)
    r_pred = torch.where(r_sim_matrix > 0.5, 1.0, 0.0)
    # s_smi_matrix_label =
    number_instance = labels.shape[0]
    top_k  = max(int(number_instance * 0.2), 1)
    label_matrix = torch.where(labels.unsqueeze(1) == labels.unsqueeze(1).T, 1.0, 0.0)
    # print("s_sim_matrix:", s_sim_matrix)
    # print("s_pred:", s_pred)
    # print("r_sim_matrix:", r_sim_matrix)
    # print("r_pred:", r_pred)
    # print("label_matrix:", label_matrix)

    neg_s_sim = torch.where(label_matrix == s_pred, 0.0, 1.0)
    neg_r_sim = torch.where(label_matrix == r_pred, 0.0, 1.0)
    #
    # print('neg_s_smi:', neg_s_sim)
    # print('neg_r_sim:', neg_r_sim)

    neg_all_sim = neg_s_sim * neg_r_sim

    # print('neg_all_sim:', neg_all_sim)

    # print('neg_all_smi_sum', torch.mean(neg_all_sim, dim=1))
    sum_neg_all_smi=torch.sum(neg_all_sim, dim=1)
    topk_values, topk_indices = torch.topk(sum_neg_all_smi, top_k, largest=True)
    mask =  sum_neg_all_smi >= topk_values[-1]
    # print(mask)

    # print(all_emb)
    hn_emb = all_emb[mask]
    # print(hn_emb)
    hn_label = labels[mask]
    # print(hn_label)
    return hn_emb,hn_label

def set_parameter_learning_rate(model,config):
    # optimization setting for bert and other parameters
    if not hasattr(model, 'bert'):
        optimizer_grouped_parameters = [{
            'params': [p for n, p in list(model.named_parameters())],
            'weight_decay': config.weight_decay,
            'lr': config.bert_learning_rate
        }]
    else:
        no_decay = ["bias", "LayerNorm.weight"]
        bert_param_optimizer = list(model.bert.named_parameters())
        bert_param_optimizer_names = [n for n, p in bert_param_optimizer]
        other_param_optimizer = [(n, p) for n, p in list(model.named_parameters())
                                 if not any(bn in n for bn in bert_param_optimizer_names)]
        other_param_optimizer_names = [n for n, p in other_param_optimizer]
        optimizer_grouped_parameters = [{
            'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
                config.weight_decay,
            'lr':
                config.bert_learning_rate
        }, {
            'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': config.bert_learning_rate
        }, {
            'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
                config.weight_decay,
            'lr':
                config.other_learning_rate
        }, {
            'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0,
            'lr': config.other_learning_rate
        }]

    return optimizer_grouped_parameters

def save_model(model, model_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def load_model(model, model_path, strict=False):
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    return model


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.enabled = False
# setup_seed(2021)
def mkdir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path,exist_ok=True)
        print("The new directory {} is created!".format(path))
#
# def mk_output_file(curFileName, config):
#     # 定义要模型产生的输出的目录
#     save_base_folder = "{}/{}".format(config['save_path'], curFileName)
#     save_checkpoints = "{}/train_checkpoints".format(save_base_folder, )
#     save_bestmodels = "{}/train_bestmodels".format(save_base_folder)
#     save_train_result = "{}/train_result".format(save_base_folder)
#     save_train_logs = "{}/train_logs".format(save_base_folder)
#     save_test_output = "{}/test_output".format(save_base_folder)
#     save_test_result = "{}/test_result".format(save_base_folder)
#     # 调用函数
#     mkdir(save_base_folder)
#     mkdir(save_checkpoints)
#     mkdir(save_bestmodels)
#     mkdir(save_train_result)
#     mkdir(save_train_logs)
#     mkdir(save_test_output)
#     mkdir(save_test_result)


def adjMatrix_to_cooMatrix(adj_matrix):
    # adj_matrix 是邻接矩阵
    # tmp_coo = sp.coo_matrix(adj_matrix)
    # values = tmp_coo.data
    # indices = np.vstack((tmp_coo.row, tmp_coo.col))
    # i = torch.LongTensor(indices)
    # v = torch.LongTensor(values)
    # edge_index = torch.sparse_coo_tensor(i, v, tmp_coo.shape)
    edge_index = adj_matrix.nonzero().t().contiguous()
    return edge_index


def draw_acc_loss_curve(resPath,savePath):

    matplotlib.use('Agg')

    data = parseJson(resPath)

    # train_res_file = parseJson(res_file_path)
    best_modelPath = data["best_model_path"]

    model_name = best_modelPath.split("/")[-1].split(".pkt")[0]
    imgPath = "{}/{}.png".format(savePath,model_name)

    train_accs = data['train_accs']
    train_pres = data['train_pres']
    train_recs = data['train_recs']
    train_f1s = data['train_f1s']
    train_loss = data['train_loss']
    eval_accs = data['eval_accs']
    eval_loss = data['eval_loss']
    eval_pres = data['eval_pres']
    eval_recs = data['eval_recs']
    eval_f1s = data['eval_f1s']

    test_metrics =  data['test_metrics'].split("\t")
    pre = test_metrics[-3]
    rec = test_metrics[-2]
    f1 = test_metrics[-1]

    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(32, 8))
    plt.subplot(1, 4, 1)
    plt.plot(epochs, train_accs, color='green', label='Training Acc')
    plt.plot(epochs, eval_accs, color='red', label='Validation Acc')
    # plt.title("ACC")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()

    plt.subplot(1, 4, 2)
    plt.plot(epochs, train_loss, color='skyblue', label='Training Loss')
    plt.plot(epochs, eval_loss, color='blue', label='Validation Loss')
    # plt.title("LOSS")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()
    plt.title(model_name + "\nP:{} R:{} F1:{}".format(pre, rec, f1))

    plt.subplot(1, 4, 3)
    plt.plot(epochs, train_pres, color='skyblue', label='Train precision')
    plt.plot(epochs, train_recs, color='blue', label='Train recall')
    plt.plot(epochs, train_f1s, color='red', label='Train f1')
    # plt.title("LOSS")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()


    plt.subplot(1, 4, 4)
    plt.plot(epochs, eval_pres, color='skyblue', label='Validation precision')
    plt.plot(epochs, eval_recs, color='blue', label='Validation recall')
    plt.plot(epochs, eval_f1s, color='red', label='Validation f1')
    # plt.title("LOSS")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()




    # plt.tight_layout()



    # plt.text(0.5, 1, test_metrics)
    # plt.text(-5, 60, 'Parabola $Y = x^2$', fontsize=22)
    plt.savefig(imgPath, dpi=120, bbox_inches='tight')  # dpi 代表像素
    # plt.show()
    plt.cla()


def draw_acc_loss_5_curve(resPath,savePath):

    matplotlib.use('Agg')

    data = parseJson(resPath)

    # train_res_file = parseJson(res_file_path)
    best_modelPath = data["best_model_path"]

    model_name = best_modelPath.split("/")[-1].split(".pkt")[0]
    imgPath = "{}/{}.png".format(savePath,model_name)

    train_accs = data['train_accs']
    train_pres = data['train_pres']
    train_recs = data['train_recs']
    train_f1s = data['train_f1s']
    train_loss = data['train_loss']
    eval_accs = data['eval_accs']
    eval_loss = data['eval_loss']
    eval_pres = data['eval_pres']
    eval_recs = data['eval_recs']
    eval_f1s = data['eval_f1s']

    test_pres = data['test_pres']
    test_recs = data['test_recs']
    test_f1s = data['test_f1s']

    test_metrics =  data['test_metrics'].split("\t")
    pre = test_metrics[-3]
    rec = test_metrics[-2]
    f1 = test_metrics[-1]

    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(40, 8))
    plt.subplot(1, 5, 1)
    plt.plot(epochs, train_accs, color='green', label='Training Acc')
    plt.plot(epochs, eval_accs, color='red', label='Validation Acc')
    # plt.title("ACC")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()

    plt.subplot(1, 5, 2)
    plt.plot(epochs, train_loss, color='skyblue', label='Training Loss')
    plt.plot(epochs, eval_loss, color='blue', label='Validation Loss')
    # plt.title("LOSS")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()
    plt.title(model_name + "\nP:{} R:{} F1:{}".format(pre, rec, f1))

    plt.subplot(1, 5, 3)
    plt.plot(epochs, train_pres, color='skyblue', label='Train precision')
    plt.plot(epochs, train_recs, color='blue', label='Train recall')
    plt.plot(epochs, train_f1s, color='red', label='Train f1')
    # plt.title("LOSS")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()


    plt.subplot(1, 5, 4)
    plt.plot(epochs, eval_pres, color='skyblue', label='Validation precision')
    plt.plot(epochs, eval_recs, color='blue', label='Validation recall')
    plt.plot(epochs, eval_f1s, color='red', label='Validation f1')
    # plt.title("LOSS")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()


    plt.subplot(1, 5, 5)
    plt.plot(epochs, test_pres, color='skyblue', label='test precision')
    plt.plot(epochs, test_recs, color='blue', label='test recall')
    plt.plot(epochs, test_f1s, color='red', label='test f1')
    # plt.title("LOSS")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()

    # plt.tight_layout()



    # plt.text(0.5, 1, test_metrics)
    # plt.text(-5, 60, 'Parabola $Y = x^2$', fontsize=22)
    plt.savefig(imgPath, dpi=120, bbox_inches='tight')  # dpi 代表像素
    # plt.show()
    plt.cla()

def draw_acc_loss_curve_pair(resPath,savePath):

    matplotlib.use('Agg')

    data = parseJson(resPath)

    # train_res_file = parseJson(res_file_path)
    best_modelPath = data["best_model_path"]

    model_name = best_modelPath.split("/")[-1].split(".pkt")[0]
    imgPath = "{}/{}.png".format(savePath,model_name)

    train_accs = data['train_accs']
    # train_pres = data['train_pres']
    # train_recs = data['train_recs']
    # train_f1s = data['train_f1s']
    train_loss = data['train_loss']
    eval_accs = data['eval_accs']
    eval_loss = data['eval_loss']
    # eval_pres = data['eval_pres']
    # eval_recs = data['eval_recs']
    # eval_f1s = data['eval_f1s']

    test_metrics =  data['test_metrics'].split("\t")
    pre = test_metrics[-3]
    rec = test_metrics[-2]
    f1 = test_metrics[-1]

    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accs, color='green', label='Training Acc')
    plt.plot(epochs, eval_accs, color='red', label='Validation Acc')
    # plt.title("ACC")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss, color='skyblue', label='Training Loss')
    plt.plot(epochs, eval_loss, color='blue', label='Validation Loss')
    # plt.title("LOSS")
    plt.legend()  # 绘制图例，默认在右上角
    plt.xlabel('Epochs')
    plt.ylabel('Rate')
    plt.grid()
    plt.title(model_name + "\nP:{} R:{} F1:{}".format(pre, rec, f1))

    # plt.subplot(1, 4, 3)
    # plt.plot(epochs, train_pres, color='skyblue', label='Train precision')
    # plt.plot(epochs, train_recs, color='blue', label='Train recall')
    # plt.plot(epochs, train_f1s, color='red', label='Train f1')
    # plt.title("LOSS")
    # plt.legend()  # 绘制图例，默认在右上角
    # plt.xlabel('Epochs')
    # plt.ylabel('Rate')
    # plt.grid()


    # plt.subplot(1, 4, 4)
    # plt.plot(epochs, eval_pres, color='skyblue', label='Validation precision')
    # plt.plot(epochs, eval_recs, color='blue', label='Validation recall')
    # plt.plot(epochs, eval_f1s, color='red', label='Validation f1')
    # # plt.title("LOSS")
    # plt.legend()  # 绘制图例，默认在右上角
    # plt.xlabel('Epochs')
    # plt.ylabel('Rate')
    # plt.grid()




    # plt.tight_layout()



    # plt.text(0.5, 1, test_metrics)
    # plt.text(-5, 60, 'Parabola $Y = x^2$', fontsize=22)
    plt.savefig(imgPath, dpi=120, bbox_inches='tight')  # dpi 代表像素
    # plt.show()
    plt.cla()


def getFileNameByPS(path, suffix):
    """

    :param path: abs path
    :param suffix: .txt
    :return:
    """
    # 获取指定目录下的所有指定后缀的文件名
    input_template_All=[]
    f_list = os.listdir(path)#返回文件名

    for i in f_list:
        # os.path.splitext():分离文件名与扩展名
        a = os.path.splitext(i)[1]
        if os.path.splitext(i)[1] == suffix:
            input_template_All.append(i)
            #print(i)
    return input_template_All

import time

def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time

@timeit
def func():
    for _ in range(3):
        time.sleep(1)

if __name__ == "__main__":
    func()