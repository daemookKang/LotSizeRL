
from Graph_make import *
from FJS_Due_Cplex_csum import schuduling2
from GNN_Train_imitation_edge_no_batch_new_graph import *
from GNN_Model_final_new_GAT2 import *
import matplotlib.pyplot as plt
def makeinputdata_dic(joblist,MLIst,done_o_lsit,done_edge_list):

    x = []
    mu = []
    O_s_list = []
    O_s = {}
    PT_list = []
    M_s_list = []
    PT_e = {}
    M_s ={}
    edge_index_P = []
    edge_index_M = []
    edge_dic = {}
    o_dic = {}
    o_dic_ind = {}
    j_dummy = joblist[0]

    index = 0

    for k in MLIst:
        x.append([index])
        mu.append([0])
        O_s.update({str(index): 1})
        O_s_list.append([1])
        o_dic.update({str(index):[j_dummy,Operation([k]),1]})
        o_dic_ind.update({j_dummy.Jtype+"_"+str(0): str(index) })
        index += 1


    for i in joblist:
        if i == j_dummy:
            continue
        for j in range(len(i.get_Sequence())):
            x.append([index])
            mu.append([0])
            if str(i.get_Index())+str(j+1) in done_o_lsit:
                #print("OK :: node")
                O_s.update({str(index): 1})
                O_s_list.append([1])
            else:
                O_s.update({str(index): 0})
                O_s_list.append([0])
            o_dic.update({str(index) :[i,i.get_Sequence()[j],j+1] })
            o_dic_ind.update({i.Jtype + "_"+str(j):str(index)})
            if j < len(i.get_Sequence())-1:
                edge_index_P.append([index,index+1])
            index += 1
    edge_ind= 0
    for i in range(index):
        for j in range(len(MLIst),index):
            if i != j:
                for k in MLIst:
                    ok = False
                    for o in o_dic[str(i)][1].MList:
                        for q in o_dic[str(j)][1].MList:
                            if k.index == o.index and k.index == q.index:
                                ok = True
                                continue
                    if ok:
                        edge_index_M.append([i,j])
                        edge_dic.update({str(i)+"_"+str(j):edge_ind})
                        edge_ind += 1

                        PT_e.update({str(i)+"_"+str(j): o_dic[str(j)][1].get_PT(o_dic[str(j)][0],k) +
                                                        k.get_ST(o_dic[str(i)][0],o_dic[str(j)][0])})

                        if PT_e[str(i)+"_"+str(j)] > 1000:
                            print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                        PT_list.append([o_dic[str(j)][1].get_PT(o_dic[str(j)][0],k) +
                                                        k.get_ST(o_dic[str(i)][0],o_dic[str(j)][0])])


                        if str(o_dic[str(j)][0].index) +str(o_dic[str(j)][2]) + "_" + str(o_dic[str(i)][0].index)  +str(o_dic[str(i)][2]) +"_"+str(k.index) in done_edge_list:
                            M_s_list.append([1])
                            M_s.update({str(i) + "_" + str(j): 1})
                            #print("OK :: edge")
                        else:
                            M_s_list.append([0])
                            M_s.update({str(i) + "_" + str(j): 0})




    x = torch.tensor(x, dtype=torch.float)
    mu = torch.tensor(mu, dtype=torch.float)
    edge_index_M = torch.tensor(edge_index_M,dtype=torch.long)
    edge_index_P = torch.tensor(edge_index_P,dtype=torch.long)

    M_s_list = torch.tensor(M_s_list, dtype=torch.float)
    O_s_list = torch.tensor(O_s_list, dtype=torch.float)
    PT_list = torch.tensor(PT_list, dtype=torch.float)

    return x,mu,edge_index_M,edge_index_P,M_s_list,O_s_list,PT_list,o_dic,o_dic_ind,edge_dic,M_s,PT_e,O_s


if __name__ == '__main__':


    ratio = []
    ratio_fin = []
    data_set = []
    num_feats = 64
    K = 4


    job_type_num = 5
    Machine_num=4
    Operation_num=[1,3]
    job_num=4



    dataset_size = 5000
    optimal = []
    csum = []
    time_opt = []
    gnn_model = []
    gnn_roll = []
    time_roll = []
    no_future = []
    ratio2 = []
    ratio3 = []


    # model = GAT1015(in_channels=in_channels, out_channels=out_channels, heads=heads, concat=concat,
    #                 negative_slope=negative_slope, dropout=dropout,
    #                 add_self_loops=add_self_loops, bias=bias, K=K, activate_func=ReLU,
    #                 graph_embedding='Attention')
    batch_size = 32
    epoch_num = 500
    in_channels = 64
    out_channels = 64
    heads = 3
    concat = True
    negative_slope = 0.2
    dropout = 0.
    add_self_loops = True
    bias = True
    model = Net_MLP(num_feats, heads, concat, negative_slope, dropout, add_self_loops, bias)

    
    #model.load_state_dict(torch.load('gatfinals_new_201120_4.pt')) #gat3
    model.load_state_dict(torch.load('gat2_201130_45413_1e-06_end_epochs.pt')) #gat2
    model.eval()
    scen_num = 100
    for sena in range(scen_num):

        Mlist, Jlist, Olist, All_job_list = make_dataset(job_type_num = job_type_num,Machine_num=Machine_num,Operation_num=Operation_num,job_num=job_num , seed = sena)
        # Mlist, Jlist, Olist, All_job_list = MLIst, joblist, Olist, joblist
        start_time = time.time()
        a = schuduling(Mlist, All_job_list, Olist, option="Cmax", m1_CTime=0, m2_CTime=0, m3_Ctime=0, m1_job1=None,
                       m2_job2=None, m3_job3=None)

        XList, CDic, XStartDic, cmax = a.solves()
        end_time = time.time()
        time_opt.append(end_time-start_time)
        #print(XStartDic)
        a2 = schuduling2(Mlist, All_job_list, Olist, option="Cmax", m1_CTime=0, m2_CTime=0, m3_Ctime=0, m1_job1=None,
                       m2_job2=None, m3_job3=None)
        XList2, CDic2, XStartDic2, cmax2 = a2.solves()

        optimal.append(cmax)
        csum.append(cmax2)

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
        roll_out = []
        cmax_roll = []

        start_time2 = time.time()
        while(True):
            min_Q_val = 99999999999999
            can_oper_ind = 0
            min_Q_val2 = 99999999999999
            list_roll = []
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
                    
                    new_Q_val =  model(g,int(o_dic_ind[str(can_oper[0].index)+str(can_oper[1]+1)]),e_fin,can_m.index,batch_size=1, Train = False,only_clast=False)

                    print(new_Q_val)
                    print(new_edge)
                    print(new_node)

                    if min_Q_val > new_Q_val:
                        if abs(new_Q_val - min_Q_val) <= 5:
                            if abs(min_Q_val2 - new_Q_val) > 5:

                                min_Q_val2 = min_Q_val

                                list_roll = [[copy.deepcopy(new_v),copy.deepcopy(fin_node),copy.deepcopy(fin_edge),copy.deepcopy(fin_job),copy.deepcopy(fin_Machine),copy.deepcopy(fin_m_ind),copy.deepcopy(fin_work_ind),
                                             copy.deepcopy(job_c_dic),copy.deepcopy(c_list),copy.deepcopy(C2_list),copy.deepcopy(now_job_list),copy.deepcopy(done_node_list),copy.deepcopy(done_edge_list),copy.deepcopy(now_oper_list),copy.deepcopy(can_working_oper)]]
                            else:
                                list_roll.append([copy.deepcopy(new_v),copy.deepcopy(fin_node),copy.deepcopy(fin_edge),copy.deepcopy(fin_job),copy.deepcopy(fin_Machine),copy.deepcopy(fin_m_ind),copy.deepcopy(fin_work_ind),
                                             copy.deepcopy(job_c_dic),copy.deepcopy(c_list),copy.deepcopy(C2_list),copy.deepcopy(now_job_list),copy.deepcopy(done_node_list),copy.deepcopy(done_edge_list),copy.deepcopy(now_oper_list),copy.deepcopy(can_working_oper)])

                        else:
                            list_roll = []

                        new_v = int(o_dic_ind[str(can_oper[0].index)+str(can_oper[1]+1)])
                        fin_node = new_node
                        fin_edge = new_edge
                        min_Q_val = new_Q_val
                        fin_job = can_oper[0]
                        fin_Machine = Mlist[can_m.index-1]
                        fin_m_ind = can_m.index-1
                        fin_work_ind = can_oper_ind

                    elif min_Q_val2 > new_Q_val and abs(new_Q_val - min_Q_val) <= 5:
                        min_Q_val2 = new_Q_val
                        list_roll.append([copy.deepcopy(int(o_dic_ind[str(can_oper[0].index)+str(can_oper[1]+1)])), copy.deepcopy(new_node), copy.deepcopy(new_edge),
                                     copy.deepcopy(can_oper[0]), copy.deepcopy(Mlist[can_m.index-1]), copy.deepcopy(can_m.index-1),
                                     copy.deepcopy(can_oper_ind),
                                     copy.deepcopy(job_c_dic), copy.deepcopy(c_list), copy.deepcopy(C2_list),
                                     copy.deepcopy(now_job_list), copy.deepcopy(done_node_list),
                                     copy.deepcopy(done_edge_list), copy.deepcopy(now_oper_list),
                                     copy.deepcopy(can_working_oper)])

                can_oper_ind+=1

            print("select machine :: ",fin_m_ind)


            c_time = max(job_c_dic[fin_job.index],c_list[fin_m_ind]) + fin_job.get_Sequence()[can_working_oper[fin_work_ind][1]].get_PT(fin_job,fin_Machine) + fin_Machine.get_ST(now_job_list[fin_m_ind],fin_job)
            job_c_dic.update({fin_job.index:c_time})
            c_list[fin_m_ind] = c_time
            C2_list[new_v] = c_time
            now_job_list[fin_m_ind] = fin_job
            print(c_list)
            done_node_list.append(fin_node)
            done_edge_list.append(fin_edge)
            print("select node :: ",fin_node)
            print("select edge :: ",fin_edge)
            print(done_edge_list)
            print(done_node_list)
            print(now_oper_list)
            now_oper_list[fin_m_ind] = fin_node
            print(now_oper_list)
            if len(list_roll) > 0:
                for rol_i in range(len(list_roll)):
                    roll_out.append(list_roll[rol_i])
            if can_working_oper[fin_work_ind][1]+1 < len(can_working_oper[fin_work_ind][0].get_Sequence()):
                can_working_oper[fin_work_ind] = [can_working_oper[fin_work_ind][0],can_working_oper[fin_work_ind][1]+1]

            else:
                del can_working_oper[fin_work_ind]

            if len(can_working_oper) == 0:
                cmax_model = max(c_list)
                print("model :: ",cmax_model)
                gnn_model.append(cmax_model)
                ratio.append(cmax_model/cmax)
                cmax_roll.append(cmax_model)
                # if cmax_model/cmax > 1.8:
                #     aaaaaa
                if cmax_model/cmax <= 1.001:
                    roll_out=[]
                break

        while(len(roll_out) > 0):


            new_v, fin_node, fin_edge, fin_job,fin_Machine, fin_m_ind, fin_work_ind, job_c_dic, c_list, C2_list,now_job_list,done_node_list, done_edge_list, now_oper_list, can_working_oper = roll_out.pop(0)
            c_time = max(job_c_dic[fin_job.index], c_list[fin_m_ind]) + fin_job.get_Sequence()[
                can_working_oper[fin_work_ind][1]].get_PT(fin_job, fin_Machine) + fin_Machine.get_ST(
                now_job_list[fin_m_ind], fin_job)
            job_c_dic.update({fin_job.index: c_time})
            c_list[fin_m_ind] = c_time
            if max(c_list) > min(cmax_roll):
                continue

            C2_list[new_v] = c_time
            now_job_list[fin_m_ind] = fin_job
            print(c_list)
            done_node_list.append(fin_node)
            done_edge_list.append(fin_edge)
            print("select node :: ", fin_node)
            print("select edge :: ", fin_edge)
            print(done_edge_list)
            print(done_node_list)
            print(now_oper_list)
            now_oper_list[fin_m_ind] = fin_node
            print(now_oper_list)

            if can_working_oper[fin_work_ind][1] + 1 < len(can_working_oper[fin_work_ind][0].get_Sequence()):
                can_working_oper[fin_work_ind] = [can_working_oper[fin_work_ind][0],
                                                  can_working_oper[fin_work_ind][1] + 1]

            else:
                del can_working_oper[fin_work_ind]

            if len(can_working_oper) == 0:
                cmax_model = max(c_list)
                cmax_roll.append(cmax_model)
                continue

            while (True):
                min_Q_val = 99999999999999
                can_oper_ind = 0
                min_Q_val2 = 99999999999999
                list_roll = []
                for can_oper in can_working_oper:
                    for can_m in can_oper[0].get_Sequence()[can_oper[1]].MList:
                        done_node_list2 = copy.deepcopy(done_node_list)
                        done_edge_list2 = copy.deepcopy(done_edge_list)
                        new_edge = str(can_oper[0].index) + str(can_oper[1] + 1) + "_" + now_oper_list[
                            can_m.index - 1] + "_" + str(can_m.index)
                        new_node = str(can_oper[0].index) + str(can_oper[1] + 1)
                        done_node_list2.append(new_node)
                        done_edge_list2.append(new_edge)
                        x, mu, edge_index_M, edge_index_P, M_s_list, O_s_list, PT_list, M_s, o_dic_ind, edge_index_M_list, not_work_j, PT_e, edge_index_M_PT_list2 = makeinputdata(
                            All_job_list, Mlist, done_node_list, done_edge_list)

                        if fir:
                            C2_list = [[0] for i in range(len(x))]
                            C2_list = torch.tensor(C2_list, dtype=torch.float)
                            fir = False

                        g = Data(x=x, edge_index=edge_index_M.t(), M_s=M_s_list, O_s=O_s_list, PT=PT_list,
                                 edge_index_P=edge_index_P.t(), mu=mu, index_m=edge_index_M_list, PT_e=PT_e,
                                 notj=not_work_j,
                                 action=None, action_pt=None, lastind=None, c=C2_list, index_PT=edge_index_M_PT_list2)

                        e = []
                        e.append([int(o_dic_ind[now_oper_list[can_m.index - 1]])])
                        e.append([int(o_dic_ind[str(can_oper[0].index) + str(can_oper[1] + 1)])])
                        e = torch.tensor(e, dtype=torch.long)
                        e_fin = []
                        e_fin.append(e)
                        e_fin.append(PT_e[new_edge])

                        new_Q_val = model(g, int(o_dic_ind[str(can_oper[0].index) + str(can_oper[1] + 1)]), e_fin,
                                          can_m.index, batch_size=1, Train=False, only_clast=False)

                        print(new_Q_val)
                        print(new_edge)
                        print(new_node)

                        if min_Q_val > new_Q_val:
                            if abs(new_Q_val - min_Q_val) <= 3:
                                if abs(min_Q_val2 - new_Q_val) > 3 :

                                    min_Q_val2 = min_Q_val

                                    list_roll = [
                                        [copy.deepcopy(new_v), copy.deepcopy(fin_node), copy.deepcopy(fin_edge),
                                         copy.deepcopy(fin_job), copy.deepcopy(fin_Machine), copy.deepcopy(fin_m_ind),
                                         copy.deepcopy(fin_work_ind),
                                         copy.deepcopy(job_c_dic), copy.deepcopy(c_list), copy.deepcopy(C2_list),
                                         copy.deepcopy(now_job_list), copy.deepcopy(done_node_list),
                                         copy.deepcopy(done_edge_list), copy.deepcopy(now_oper_list),
                                         copy.deepcopy(can_working_oper)]]
                                else:
                                    list_roll.append(
                                        [copy.deepcopy(new_v), copy.deepcopy(fin_node), copy.deepcopy(fin_edge),
                                         copy.deepcopy(fin_job), copy.deepcopy(fin_Machine), copy.deepcopy(fin_m_ind),
                                         copy.deepcopy(fin_work_ind),
                                         copy.deepcopy(job_c_dic), copy.deepcopy(c_list), copy.deepcopy(C2_list),
                                         copy.deepcopy(now_job_list), copy.deepcopy(done_node_list),
                                         copy.deepcopy(done_edge_list), copy.deepcopy(now_oper_list),
                                         copy.deepcopy(can_working_oper)])

                            else:
                                list_roll = []

                            new_v = int(o_dic_ind[str(can_oper[0].index) + str(can_oper[1] + 1)])
                            fin_node = new_node
                            fin_edge = new_edge
                            min_Q_val = new_Q_val
                            fin_job = can_oper[0]
                            fin_Machine = Mlist[can_m.index - 1]
                            fin_m_ind = can_m.index - 1
                            fin_work_ind = can_oper_ind

                        elif min_Q_val2 > new_Q_val and abs(new_Q_val - min_Q_val) <= 3:
                            min_Q_val2 = new_Q_val
                            list_roll.append(
                                [copy.deepcopy(int(o_dic_ind[str(can_oper[0].index) + str(can_oper[1] + 1)])),
                                 copy.deepcopy(new_node), copy.deepcopy(new_edge),
                                 copy.deepcopy(can_oper[0]), copy.deepcopy(Mlist[can_m.index - 1]),
                                 copy.deepcopy(can_m.index - 1),
                                 copy.deepcopy(can_oper_ind),
                                 copy.deepcopy(job_c_dic), copy.deepcopy(c_list), copy.deepcopy(C2_list),
                                 copy.deepcopy(now_job_list), copy.deepcopy(done_node_list),
                                 copy.deepcopy(done_edge_list), copy.deepcopy(now_oper_list),
                                 copy.deepcopy(can_working_oper)])

                    can_oper_ind += 1

                print("select machine :: ", fin_m_ind)

                c_time = max(job_c_dic[fin_job.index], c_list[fin_m_ind]) + fin_job.get_Sequence()[
                    can_working_oper[fin_work_ind][1]].get_PT(fin_job, fin_Machine) + fin_Machine.get_ST(
                    now_job_list[fin_m_ind], fin_job)
                job_c_dic.update({fin_job.index: c_time})
                c_list[fin_m_ind] = c_time
                if max(c_list) > min(cmax_roll):
                    break
                C2_list[new_v] = c_time
                now_job_list[fin_m_ind] = fin_job
                print(c_list)
                done_node_list.append(fin_node)
                done_edge_list.append(fin_edge)
                print("select node :: ", fin_node)
                print("select edge :: ", fin_edge)
                print(done_edge_list)
                print(done_node_list)
                print(now_oper_list)
                now_oper_list[fin_m_ind] = fin_node
                print(now_oper_list)
                if len(list_roll) > 0:
                    for rol_i in range(len(list_roll)):
                        roll_out.append(list_roll[rol_i])
                if can_working_oper[fin_work_ind][1] + 1 < len(can_working_oper[fin_work_ind][0].get_Sequence()):
                    can_working_oper[fin_work_ind] = [can_working_oper[fin_work_ind][0],
                                                      can_working_oper[fin_work_ind][1] + 1]

                else:
                    del can_working_oper[fin_work_ind]

                if len(can_working_oper) == 0:
                    cmax_model = max(c_list)
                    cmax_roll.append(cmax_model)
                    # if cmax_model/cmax > 1.8:
                    #     aaaaaa
                    if cmax_model / cmax <= 1.001:
                        roll_out = []
                    break

        gnn_roll.append(min(cmax_roll))
        ratio3.append(min(cmax_roll)/cmax)
        end_time2 = time.time()
        time_roll.append(end_time2-start_time2)


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
                job_c_dic.update({i.index: 0})
                can_working_oper.append([i, 0])
        fir = True
        while (True):
            min_Q_val = 99999999999999
            can_oper_ind = 0
            for can_oper in can_working_oper:

                for can_m in can_oper[0].get_Sequence()[can_oper[1]].MList:
                    done_node_list2 = copy.deepcopy(done_node_list)
                    done_edge_list2 = copy.deepcopy(done_edge_list)
                    new_edge = str(can_oper[0].index) + str(can_oper[1] + 1) + "_" + now_oper_list[
                        can_m.index - 1] + "_" + str(can_m.index)
                    new_node = str(can_oper[0].index) + str(can_oper[1] + 1)
                    done_node_list2.append(new_node)
                    done_edge_list2.append(new_edge)
                    x, mu, edge_index_M, edge_index_P, M_s_list, O_s_list, PT_list, M_s, o_dic_ind, edge_index_M_list, not_work_j, PT_e, edge_index_M_PT_list2 = makeinputdata(
                        All_job_list, Mlist, done_node_list, done_edge_list)

                    if fir:
                        C2_list = [[0] for i in range(len(x))]
                        C2_list = torch.tensor(C2_list, dtype=torch.float)
                        fir = False

                    g = Data(x=x, edge_index=edge_index_M.t(), M_s=M_s_list, O_s=O_s_list, PT=PT_list,
                             edge_index_P=edge_index_P.t(), mu=mu, index_m=edge_index_M_list, PT_e=PT_e,
                             notj=not_work_j,
                             action=None, action_pt=None, lastind=None, c=C2_list, index_PT=edge_index_M_PT_list2)

                    e = []
                    e.append([int(o_dic_ind[now_oper_list[can_m.index - 1]])])
                    e.append([int(o_dic_ind[str(can_oper[0].index) + str(can_oper[1] + 1)])])
                    e = torch.tensor(e, dtype=torch.long)
                    e_fin = []
                    e_fin.append(e)
                    e_fin.append(PT_e[new_edge])

                    new_Q_val = model(g, int(o_dic_ind[str(can_oper[0].index) + str(can_oper[1] + 1)]), e_fin,
                                      can_m.index, batch_size=1, Train = False,only_clast=True)

                    print(new_Q_val)
                    print(new_edge)
                    print(new_node)

                    if min_Q_val > new_Q_val:
                        new_v = int(o_dic_ind[str(can_oper[0].index) + str(can_oper[1] + 1)])
                        fin_node = new_node
                        fin_edge = new_edge
                        min_Q_val = new_Q_val
                        fin_job = can_oper[0]
                        fin_Machine = Mlist[can_m.index - 1]
                        fin_m_ind = can_m.index - 1
                        fin_work_ind = can_oper_ind

                can_oper_ind += 1

            print("select machine :: ", fin_m_ind)

            c_time = max(job_c_dic[fin_job.index], c_list[fin_m_ind]) + fin_job.get_Sequence()[
                can_working_oper[fin_work_ind][1]].get_PT(fin_job, fin_Machine) + fin_Machine.get_ST(
                now_job_list[fin_m_ind], fin_job)
            job_c_dic.update({fin_job.index: c_time})
            c_list[fin_m_ind] = c_time
            C2_list[new_v] = c_time
            now_job_list[fin_m_ind] = fin_job
            print(c_list)
            done_node_list.append(fin_node)
            done_edge_list.append(fin_edge)
            print("select node :: ", fin_node)
            print("select edge :: ", fin_edge)
            print(done_edge_list)
            print(done_node_list)
            print(now_oper_list)
            now_oper_list[fin_m_ind] = fin_node
            print(now_oper_list)
            if can_working_oper[fin_work_ind][1] + 1 < len(can_working_oper[fin_work_ind][0].get_Sequence()):
                can_working_oper[fin_work_ind] = [can_working_oper[fin_work_ind][0],
                                                  can_working_oper[fin_work_ind][1] + 1]

            else:
                del can_working_oper[fin_work_ind]

            if len(can_working_oper) == 0:
                cmax_model = max(c_list)
                print("model :: ", cmax_model)
                no_future.append(cmax_model)
                ratio2.append(cmax_model / cmax)
                final_ra = min(no_future[-1],gnn_model[-1])
                ratio_fin.append(final_ra/cmax)
                break
        # if cmax_model/cmax > 1.7:
        #     break
    gnn_model = np.array(gnn_model)
    optimal = np.array(optimal)
    gnn_roll = np.array(gnn_roll)

    csum = np.array(csum)
    ratio =np.array(ratio)
    ratio2 = np.array(ratio2)
    ratio3 = np.array(ratio3)
    ratio_fin = np.array(ratio_fin)
    time_roll = np.array(time_roll)
    time_opt = np.array(time_opt)
    no_future = np.array(no_future)


    plt.figure()
    # plt.plot(list(range(100)), gnn_model, label='gnn')
    # plt.plot(list(range(100)), optimal, label='opt')
    plt.plot(list(range(scen_num)), ratio, label='gnn')
    plt.plot(list(range(scen_num)), ratio2, label='earliest ctime')
    plt.plot(list(range(scen_num)), ratio3, label='rollout_step3')
    print("opt mean :: ",optimal.mean())
    print("csum mean :: ",csum.mean())
    print("earilist mean :: ", no_future.mean())
    print("csum ratio ::",csum.mean()/optimal.mean())
    print("model mean :: ",gnn_model.mean())
    print("roll out mean :: ", gnn_roll.mean())
    print("opt mean time :: ",time_opt.mean())
    print("rollout mean time :: ", time_roll.mean())

    print("ratio mean :: ", ratio.mean())
    print("ratio2 mean :: ", ratio2.mean())
    print("roll_ratio mean :; ",ratio3.mean())
    print("ratiofin mean :: ", ratio_fin.mean())
    print(ratio)
    print(ratio2)
    print(ratio3)

    plt.legend()