
from Graph_make import *
from GNN_Train_imitation_edge_no_batch_new import *
from GNN_Model_final_new_GAT2 import *
import matplotlib.pyplot as plt


if __name__ == '__main__':

    ratio = []
    ratio_fin = []
    data_set = []
    num_feats = 64
    K = 4







    dataset_size = 32



    # model = GAT1015(in_channels=in_channels, out_channels=out_channels, heads=heads, concat=concat,
    #                 negative_slope=negative_slope, dropout=dropout,
    #                 add_self_loops=add_self_loops, bias=bias, K=K, activate_func=ReLU,
    #                 graph_embedding='Attention')
    batch_size = 32
    epoch_num = 10000000
    heads = 3
    concat = True
    negative_slope = 0.2
    dropout = 0.
    add_self_loops = True
    bias = True
    model = Net_MLP(num_feats, heads, concat, negative_slope, dropout, add_self_loops, bias)

    model.load_state_dict(torch.load('gatfinals22_201127_3.pt'))

    scen_num = 100

    learning_rate = 0.000001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = nn.MSELoss
    train_loss_lst = [[] for _ in range(2000)]
    for k in range(2000):
        gvemy_list = []
        for sena in range(epoch_num):
            model.eval()
            job_type_num = random.randint(3, 8)
            Machine_num = random.randint(2, 7)
            Operation_num = [random.randint(1, 2), random.randint(2, 5)]
            job_num = random.randint(3, 8)

            job_type_num = 5
            Machine_num = 4
            Operation_num = [1, 3]
            job_num = 4


            scena = random.randint(0, 10000)
            # print()


            Mlist, Jlist, Olist, All_job_list = make_dataset(job_type_num = job_type_num,Machine_num=Machine_num,Operation_num=Operation_num,job_num=job_num  )
            # Mlist, Jlist, Olist, All_job_list = MLIst, joblist, Olist, joblist

            #print(XStartDic)




            done_node_list = []
            done_edge_list = []
            Machine_num = len(Mlist)
            c_list = [0 for i in range(Machine_num)]

            now_oper_list = ["01" for i in range(Machine_num)]
            now_job_list = [All_job_list[0] for i in range(Machine_num)]
            job_c_dic = {}
            can_working_oper = []

            for i in All_job_list:
                if i.index != 0:
                    job_c_dic.update({i.index:0})
                    can_working_oper.append([i, 0])
            fir = True

            g_list = []
            v_list = []
            e_fin_list = []
            ma_list = []
            cs_list = []

            while(True):
                min_Q_val = 99999999999999
                can_oper_ind = 0
                for can_oper in can_working_oper:

                    for can_m in can_oper[0].get_Sequence()[can_oper[1]].MList:
                        done_node_list2 = copy.deepcopy(done_node_list)
                        done_edge_list2 = copy.deepcopy(done_edge_list)
                        new_edge = str(can_oper[0].index)+str(can_oper[1]+1)+"_" +now_oper_list[can_m.index-1]+"_"+str(can_m.index)
                        new_node = str(can_oper[0].index)+str(can_oper[1]+1)
                        done_node_list2.append(new_node)
                        done_edge_list2.append(new_edge)
                        x, mu, edge_index_M, edge_index_P, M_s_list, O_s_list, PT_list, M_s, o_dic_ind, edge_index_M_list, not_work_j, PT_e, edge_index_M_PT_list2 = makeinputdata(
                            All_job_list, Mlist,done_node_list, done_edge_list)

                        if fir:
                            C2_list = [[0] for i in range(len(x))]
                            C2_list = torch.tensor(C2_list, dtype=torch.float)
                            fir = False

                        g = Data(x=x, edge_index=edge_index_M.t(), M_s=M_s_list, O_s=O_s_list, PT=PT_list,
                     edge_index_P=edge_index_P.t(), mu=mu, index_m=edge_index_M_list, PT_e=PT_e, notj=not_work_j,
                     action=None, action_pt=None , lastind= None , c =C2_list,index_PT=edge_index_M_PT_list2)


                        e = []
                        e.append([int(o_dic_ind[now_oper_list[can_m.index-1]])])
                        e.append([int(o_dic_ind[str(can_oper[0].index)+str(can_oper[1]+1)])])
                        e = torch.tensor(e, dtype=torch.long)
                        e_fin = []
                        e_fin.append(e)
                        e_fin.append(PT_e[new_edge])

                        new_Q_val =  model(g,int(o_dic_ind[str(can_oper[0].index)+str(can_oper[1]+1)]),e_fin,can_m.index,batch_size=1,Train=False)

                        # print(new_Q_val)
                        # print(new_edge)
                        # print(new_node)

                        if min_Q_val > new_Q_val:
                            new_v = int(o_dic_ind[str(can_oper[0].index)+str(can_oper[1]+1)])
                            fin_node = new_node
                            fin_edge = new_edge
                            min_Q_val = new_Q_val
                            fin_job = can_oper[0]
                            fin_Machine = Mlist[can_m.index-1]
                            fin_m_ind = can_m.index-1
                            fin_work_ind = can_oper_ind
                            new_g = g.clone()
                            new_e_fin = copy.deepcopy(e_fin)
                            new_m = can_m.index

                    can_oper_ind+=1

                #print("select machine :: ",fin_m_ind)


                c_time = max(job_c_dic[fin_job.index],c_list[fin_m_ind]) + fin_job.get_Sequence()[can_working_oper[fin_work_ind][1]].get_PT(fin_job,fin_Machine) + fin_Machine.get_ST(now_job_list[fin_m_ind],fin_job)
                job_c_dic.update({fin_job.index:c_time})
                c_list[fin_m_ind] = c_time
                C2_list[new_v] = c_time
                now_job_list[fin_m_ind] = fin_job
                # print(c_list)
                done_node_list.append(fin_node)
                done_edge_list.append(fin_edge)
                # print("select node :: ",fin_node)
                # print("select edge :: ",fin_edge)
                # print(done_edge_list)
                # print(done_node_list)
                # print(now_oper_list)
                now_oper_list[fin_m_ind] = fin_node
                # print(now_oper_list)

                g_list.append(new_g)
                v_list.append(new_v)
                e_fin_list.append(new_e_fin)
                ma_list.append(new_m)
                cs_list.append(c_time)

                # print(now_oper_list)
                if can_working_oper[fin_work_ind][1]+1 < len(can_working_oper[fin_work_ind][0].get_Sequence()):
                    can_working_oper[fin_work_ind] = [can_working_oper[fin_work_ind][0],can_working_oper[fin_work_ind][1]+1]

                else:
                    del can_working_oper[fin_work_ind]

                if len(can_working_oper) == 0:
                    for index in range(len(g_list)):

                        gvemy_list.append([g_list[index],v_list[index],e_fin_list[index],ma_list[index],c_list[ma_list[index]-1]-cs_list[index]])
                    # print("model :: ",cmax_model)


                    break

            print(len(gvemy_list))
            if dataset_size < len(gvemy_list):
                break





        model.train()
        loss_mean = 0
        tn = 0
        target_list = list()
        q_list = list()

        for data in gvemy_list:
            optimizer.zero_grad()
            target = torch.tensor(data[4])
            q_list.append(model(data[0], data[1], data[2], data[3], batch_size=1, Train=True))
            target_list.append(target)
            if len(target_list) >= 32:
                q_list = torch.cat(q_list)
                target_list = torch.tensor(target_list, dtype=torch.float)
                # print(q_list)
                # print(target_list)
                # print(target_list)
                # target_list = torch.cat(target_list)
                print(q_list, "q_list~~~~~~~~~~~~")
                train_loss = loss_func()(q_list, target_list)
                # print(Q(data, batch_size=batch_size))
                # print(data.y,"!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                train_loss.backward()
                train_loss_lst[k].append(train_loss.item())
                loss_mean += train_loss.item()
                tn += 1
                optimizer.step()
                target_list = list()
                q_list = list()

                print(train_loss)
        print("mean :: ", loss_mean/tn)

        if k != 0 and k % 200 == 0:
            torch.save(model.state_dict(), 'gat2_201130_45413_{}_{}epochs.pt'.format(learning_rate, k))
            # if cmax_model/cmax > 1.7:
            #     break
        # with open("datafinal1118_edge_{}_Rnd_ds{}.pickle".format(k,dataset_size),"wb") as fw:
        #     pickle.dump(gvemy_list,fw)

    torch.save(model.state_dict(), 'gat2_201130_45413_{}_end_epochs.pt'.format(learning_rate))
    # plt.plot(list(range(len(train_loss_lst))), train_loss_lst, label='ratio')