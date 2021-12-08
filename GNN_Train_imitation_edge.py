import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch.nn import Sequential as Seq, Linear, ReLU, ReLU6
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam, RMSprop
from collections import OrderedDict
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import time
import random
import numpy as np
from scipy.stats import pearsonr
from GNN_Model_final import *
from Graph_make import *

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def make_dataloader(dataset, batch_size = 32):
    data_list = []
    for data in dataset:
        e = [int(data[1][1][0]), int(data[1][1][1])]
        e = torch.tensor(e, dtype=torch.long)
        v = int(data[1][0])
        e_fin = []
        e_key = data[1][2]
        e_fin.append(e)


        x,mu,edge_index_M,edge_index_P,M_s_list,O_s_list,PT_list,edge_index_M_list,not_work_j,PT_e = data[0]

        g = Data(x=x, edge_index=edge_index_M.t(), M_s=M_s_list, O_s=O_s_list, PT=PT_list, edge_index_P=edge_index_P.t(),mu=mu,index_m=edge_index_M_list,PT_e=PT_e,notj = not_work_j,action=e.t(),action_pt = PT_e[e_key] )
        x2,mu2,edge_index_M2,edge_index_P2,M_s_list2,O_s_list2,PT_list2,edge_index_M_list2,not_work_j2,PT_e2= data[2]
        new_list = []
        #print(edge_index_M.t())
        g_y = Data(x=x2, edge_index=edge_index_M2.t(), M_s=M_s_list2, O_s=O_s_list2, PT=PT_list2, edge_index_P=edge_index_P2.t(),mu=mu,index_m=edge_index_M_list,PT_e=PT_e2,notj = not_work_j2,action=e.t(),action_pt = PT_e[e_key])

        e_fin.append(PT_e[e_key])


        r = torch.tensor(data[3], dtype=torch.float)
        y = torch.tensor(data[5], dtype=torch.float)
        data_list.append([g,g_y,r,y,v,e_fin])

    random.shuffle(data_list)

    train_data_ratio = 0.78
    valid_data_ratio = 0.2
    test_data_ratio = 0.02
    last_train_idx = int(len(data_list) * train_data_ratio)
    last_valid_idx = int(len(data_list) * (train_data_ratio + valid_data_ratio))
    train_loader = DataLoader(data_list[:last_train_idx], batch_size=batch_size)
    valid_loader = DataLoader(data_list[last_train_idx:last_valid_idx], batch_size=batch_size)
    test_loader = DataLoader(data_list[last_valid_idx:], batch_size=1)

    return train_loader, valid_loader, test_loader


def train(model, train_loader, valid_loader, epoch_num=1000, batch_size=32, opt=torch.optim.Adam, loss_func=nn.MSELoss):
    train_loss_lst = [[] for _ in range(epoch_num)]
    valid_loss_lst = [[] for _ in range(epoch_num)]
    Q = model
    gamma = 0.3
    # Q = model().to(device)
    optimizer = opt(Q.parameters(), lr=0.0001)
    Q.train()
    start_time = time.process_time()
    for epoch in range(epoch_num):
        print("epoch :: ",epoch)
        loss_mean = 0
        tn = 0
        for data in train_loader:
            if len(data[2]) == batch_size:

                optimizer.zero_grad()
                #print(len(Q(data[0], batch_size=batch_size)))
                #print(Q(data[1], batch_size=batch_size))
                # print(len(Q(data[1], batch_size=batch_size)))
                #print(gamma*Q(data[1], batch_size=batch_size))
                # print(len(Q(data[0], batch_size=batch_size)))
                if True:
                    target = data[3]
                else:
                    target = data[2]+gamma*Q(data[1],data[4], batch_size=batch_size)
                train_loss = loss_func()(Q(data[0],data[4],data[5], batch_size=batch_size), target)
                # print(Q(data, batch_size=batch_size))
                # print(data.y,"!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                train_loss.backward()
                train_loss_lst[epoch].append(train_loss.item())
                loss_mean += train_loss.item()
                tn += 1
                optimizer.step()
                print(train_loss)
        print("mean :: ",loss_mean/tn)
        for data in valid_loader:
            if len(data[2]) == batch_size:
                if True:
                    target = data[3]
                else:
                    target = data[2] + gamma * Q(data[1],data[4], batch_size=batch_size)

                test_loss = nn.MSELoss()(Q(data[0],data[4],data[5], batch_size=batch_size), target)
                valid_loss_lst[epoch].append(test_loss.item())
        if epoch % 50 == 0 and epoch > 1:
            torch.save(Q.state_dict(), 'mlpfinals_201111_2_{}epochs.pt'.format(epoch))
    end_time = time.process_time()
    print(f'Training time: {end_time - start_time} sec')
    return Q, train_loss_lst, valid_loss_lst



def plot_total_train_loss(train_loss_lst):
    tot_train_loss_lst = []
    for loss_lst in train_loss_lst:
        tot_train_loss_lst += loss_lst
    plt.figure()
    plt.plot(list(range(len(tot_train_loss_lst))), tot_train_loss_lst)


def plot_avg_train_valid_loss(train_loss_lst, test_loss_lst):
    epoch_num = len(train_loss_lst)
    train_loss_by_epoch, test_loss_by_epoch = [], []
    for epoch in range(epoch_num):
        train_loss_by_epoch.append(np.mean(train_loss_lst[epoch]))
        test_loss_by_epoch.append(np.mean(test_loss_lst[epoch]))
    plt.figure()
    plt.plot(list(range(epoch_num)), train_loss_by_epoch, label='train')
    plt.plot(list(range(epoch_num)), test_loss_by_epoch, label='valid')
    plt.legend()


def test(test_loader, model, j=0):
    y_lst, pred_y_lst = [], []
    for data in test_loader:
        if j > 0:
            print(len(data['x']))
            print(data['M'])
            print(data['R'])
            print('number of stage', int(len(data['x']) - data['M'].item() - data['R'].item()))
            if int(len(data['x']) - data['M'].item() - data['R'].item()) == j:
                y = data['y']
                pred_y = model(data, batch_size=1)
                print(pred_y.item(), y)
                y_lst.append(y.item())
                pred_y_lst.append(pred_y.item())
        else:
            y = data['y']
            pred_y = model(data, batch_size=1)
            print(pred_y.item(), y)
            y_lst.append(y.item())
            pred_y_lst.append(pred_y.item())
    R, p_value = pearsonr(y_lst, pred_y_lst)
    # print(R)

    plt.figure()
    plt.plot(y_lst, pred_y_lst, '.')
    plt.xlabel('True value')
    plt.ylabel('Predicted value')
    return R



if __name__ == '__main__':
    num_feats = 64
    K = 4

    optimizer = torch.optim.Adam
    opt_name = 'Adam'
    learning_rate = 0.0001
    loss_func = nn.MSELoss
    lf_name = 'MSE'

    batch_size = 32
    epoch_num = 500
    in_channels = 64
    out_channels = 64
    heads = 2
    concat = True
    negative_slope = 0.2
    dropout = 0.
    add_self_loops = True
    bias = True
    graph_embedding = 'Attention'

    # model = GAT1015(in_channels=in_channels, out_channels=out_channels, heads=heads, concat=concat,
    #                 negative_slope=negative_slope, dropout=dropout,
    #                 add_self_loops=add_self_loops, bias=bias, K=K, activate_func=ReLU,
    #                 graph_embedding='Attention').to(device)
    model = Net_MLP(num_feats).to(device)
    with open("datafinal2_edge_SARS_j4_o[2, 3]_m3_jt5_gamma0.5_ds10000.pickle", "rb") as fr:
        dataset = pickle.load(fr)

    # with open("data_SARS_j5_o[1, 3]_m4_jt5_ds5000.pickle", "rb") as fr:
    #     dataset2 = pickle.load(fr)
    #
    #
    # dataset.extend(dataset2)


    train_loader, valid_loader, test_loader = make_dataloader(dataset, batch_size)
    Q, train_loss_lst, valid_loss_lst = train(model, train_loader, valid_loader, epoch_num=500, batch_size=32, opt=torch.optim.Adam, loss_func=nn.MSELoss)
    torch.save(Q.state_dict(), 'mlpfinals_201111_2_.pt')
    plot_total_train_loss(train_loss_lst)
    plot_avg_train_valid_loss(train_loss_lst,valid_loss_lst)