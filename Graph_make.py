import pickle
import random
from random import randint
import copy
import numpy as np
# from MIP_FJS import schuduling
from FJS_Due_Cplex import schuduling

import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data

import networkx as nx
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt

class Job:
    def __init__(self, sequence, index, release_date, Jtype, dueDate=None):
        self.sequence = sequence
        self.index = index
        self.realseDate = release_date
        self.Jtype = Jtype
        self.stateList = []
        self.dueDate = dueDate
        for i in range(len(self.sequence)):
            self.stateList.append("yet")

    def get_Jtype(self):
        return self.Jtype

    def set_Jtype(self, Jtype):
        self.Jtype = Jtype

    def get_realseDate(self):
        return self.realseDate

    def set_realseDate(self, releaseDate):
        self.realseDate = releaseDate

    def get_Index(self):
        return self.index

    def get_Sequence(self):
        return self.sequence

    def set_Index(self, index):
        self.index = index

    def set_Sequence(self, sequence):
        self.sequence = sequence


class Operation:
    def __init__(self, MList):
        self.MList = MList
        self.PTDic = {}
        self.MindexList = []
        for i in self.MList:
            self.MindexList.append(i.get_Index())

    def setPT(self, Job, Machine, PT):
        if Machine not in self.MList:
            return "Error M not in MList"
        else:
            self.PTDic.update({str(Job.Jtype) + str(Machine.index): PT})

    def get_PT(self, Job, Machine):
        if str(Job.Jtype) + str(Machine.index) in self.PTDic.keys():

            return self.PTDic[str(Job.Jtype) + str(Machine.index)]
        else:
            #print("PT Error : "+str(Job.Jtype)+str(Machine.index))
            return 55555555


class Machine:
    def __init__(self, index):
        self.index = index
        self.STDic = {}

    def set_ST(self, Job1, Job2, ST):
        self.STDic.update({str(Job1.Jtype) + str(Job2.Jtype): ST})

    def get_Index(self):
        return self.index

    def set_Index(self, index):
        self.index = index

    def get_ST(self, Job1, Job2):
        if str(Job1.get_Jtype()) + str(Job2.get_Jtype()) in self.STDic.keys():
            return self.STDic[str(Job1.get_Jtype()) + str(Job2.get_Jtype())]
        else:
            #print("ST Error : "+str(Job1.get_Jtype())+str(Job2.get_Jtype()))
            return 2222222



def make_dataset(job_type_num = 5,Machine_num=3,Operation_num=[1,3],job_num=5 , seed = None):
    Mlist = []
    Olist = []
    Jlist = []
    All_job_list = []
    j_machine_list = {}
    if seed != None:
        random.seed(seed)

    for i in range(Machine_num):
        Mlist.append(Machine(i+1))



    od = Operation(Mlist)
    j_dummy = Job([od], 0, 0, 'a')


    for i in range(Operation_num[1]):
        m_num = random.randint(1,Machine_num)
        Olist.append(Operation(random.sample(Mlist,m_num)))

    for i in range(job_type_num):
        jm_list = []
        o_num = random.randint(Operation_num[0], Operation_num[1])

        seq = random.sample(Olist,o_num)
        Jlist.append(Job(seq,i+1,0,chr(98+i)))
        # print("Job_type :: ",Jlist[i].Jtype," sequence :: ")


        for j in seq:
            # for k in j.MList:
            # #     print(k.index)
            # # print("______________________________________")
            jm_list.extend(j.MList)

        j_machine_list.update({chr(98+i):jm_list})



    for i in Jlist:
        for j in i.get_Sequence():
            for k in j.MList:
                pt_time = random.randint(1, 20)

                j.setPT(i,k,pt_time)


    for m in Mlist:
        for i in Jlist:
            for j in Jlist:
                if m in j_machine_list[i.Jtype] and m in j_machine_list[j.Jtype]:
                    if i == j:
                        m.set_ST(i,j,0)
                    else:
                        st_time = random.randint(1, 20)
                        m.set_ST(i, j, st_time)

    for m in Mlist:
        m.set_ST(j_dummy, j_dummy, 0)
        od.setPT(j_dummy,m,0)
        for i in Jlist:
            od.setPT(i, m, 0)
            if m in j_machine_list[i.Jtype]:
                st_time = random.randint(1, 20)
                m.set_ST(j_dummy,i,st_time)


    ind = 1
    for i in range(job_num):
        new_job = copy.deepcopy(random.sample(Jlist,1)[0])
        # new_job = Jlist[1]
        #print(new_job.Jtype)

        ind += 1
        All_job_list.append(Job(copy.deepcopy(new_job.get_Sequence()), ind, 0, copy.deepcopy(new_job.Jtype)))


    All_job_list.insert(0,j_dummy)
    Jlist.insert(0,j_dummy)
    Olist.insert(0,od)


    return Mlist, Jlist, Olist, All_job_list






def makegraph(joblist,MLIst):

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
        o_dic.update({str(index):[j_dummy,Operation([k])]})
        o_dic_ind.update({j_dummy.Jtype+"_"+str(0): str(index) })
        index += 1


    for i in joblist:
        if i == j_dummy:
            continue
        for j in range(len(i.get_Sequence())):
            x.append([index])
            mu.append([0])
            O_s.update({str(index): 0})
            O_s_list.append([0])
            o_dic.update({str(index) :[i,i.get_Sequence()[j]] })
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
                        M_s.update({str(i)+"_"+str(j):0})
                        PT_e.update({str(i)+"_"+str(j): o_dic[str(j)][1].get_PT(o_dic[str(j)][0],k) +
                                                        k.get_ST(o_dic[str(i)][0],o_dic[str(j)][0])})
                        if PT_e[str(i)+"_"+str(j)] > 1000:
                            print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                        PT_list.append([o_dic[str(j)][1].get_PT(o_dic[str(j)][0],k) +
                                                        k.get_ST(o_dic[str(i)][0],o_dic[str(j)][0])])
                        M_s_list.append([0])





    x = torch.tensor(x, dtype=torch.float)
    mu = torch.tensor(mu, dtype=torch.float)
    edge_index_M = torch.tensor(edge_index_M,dtype=torch.long)
    edge_index_P = torch.tensor(edge_index_P,dtype=torch.long)

    M_s_list = torch.tensor(M_s_list, dtype=torch.float)
    O_s_list = torch.tensor(O_s_list, dtype=torch.float)
    PT_list = torch.tensor(PT_list, dtype=torch.float)




    graph = Data(x=x, edge_index=edge_index_M.t(), M_s=M_s_list, O_s=O_s_list,PT = PT_list , edge_index_P=edge_index_P.t(), mu=mu)
    return graph

# Mlist, Jlist, Olist, All_job_list = make_dataset(job_type_num = 7,Machine_num=5,Operation_num=[2,5],job_num=15)
#
#
#
# graph = makegraph(All_job_list,Mlist)
#
# nx_graph = to_networkx(graph)
# colors = ['dodgerblue']*len(graph.x)
#
# nx.draw(nx_graph, cmap=plt.get_cmap('Set1'), node_color=colors, node_size=75, linewidths=6)
# plt.show()

def makeinputdata(joblist,MLIst,done_o_lsit,done_edge_list):

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
    edge_index_M_list=[]
    edge_index_M_PT_list= []
    edge_dic = {}
    o_dic = {}
    o_dic_ind = {}
    not_work_j = []

    j_dummy = joblist[0]

    index = 0

    for k in MLIst:
        edge_index_M_list.append([])
        edge_index_M_PT_list.append([])
        x.append([index])
        mu.append([0])
        O_s.update({str(index): 1})
        O_s_list.append([1])
        not_work_j.append([0])
        o_dic.update({str(index):[j_dummy,Operation([k]),1]})
        o_dic_ind.update({str(0)+str(k.index): str(index)})
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
                not_work_j.append([0])
            else:
                O_s.update({str(index): 0})
                O_s_list.append([0])
                not_work_j.append([1])
            o_dic.update({str(index) :[i,i.get_Sequence()[j],j+1] })
            o_dic_ind.update({str(i.index) +str(j+1):str(index)})
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
                        edge_index_M_list[k.index-1].append([i,j])
                        edge_index_M_PT_list[k.index-1].append([o_dic[str(j)][1].get_PT(o_dic[str(j)][0],k) +k.get_ST(o_dic[str(i)][0],o_dic[str(j)][0])])
                        edge_dic.update({str(i)+"_"+str(j):edge_ind})
                        edge_ind += 1

                        PT_e.update({str(o_dic[str(j)][0].index) +str(o_dic[str(j)][2]) + "_" + str(o_dic[str(i)][0].index)  + str(o_dic[str(i)][2]) +"_"+str(k.index)
                                     : o_dic[str(j)][1].get_PT(o_dic[str(j)][0],k) +k.get_ST(o_dic[str(i)][0],o_dic[str(j)][0])})

                        if PT_e[str(o_dic[str(j)][0].index) +str(o_dic[str(j)][2]) + "_" + str(o_dic[str(i)][0].index)  + str(o_dic[str(i)][2]) +"_"+str(k.index)] > 1000:
                            print("ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

                        PT_list.append([o_dic[str(j)][1].get_PT(o_dic[str(j)][0],k) +
                                                        k.get_ST(o_dic[str(i)][0],o_dic[str(j)][0])])


                        if str(o_dic[str(j)][0].index) +str(o_dic[str(j)][2]) + "_" + str(o_dic[str(i)][0].index)  + str(o_dic[str(i)][2]) +"_"+str(k.index) in done_edge_list:
                            M_s_list.append([1])
                            M_s.update({str(i) + str(j): 1})
                            #print("OK :: edge")
                        else:
                            M_s_list.append([0])
                            M_s.update({str(i)  + str(j): 0})




    x = torch.tensor(x, dtype=torch.float)
    mu = torch.tensor(mu, dtype=torch.float)

    edge_index_M = torch.tensor(edge_index_M,dtype=torch.long)
    edge_index_P = torch.tensor(edge_index_P,dtype=torch.long)

    M_s_list = torch.tensor(M_s_list, dtype=torch.float)
    O_s_list = torch.tensor(O_s_list, dtype=torch.float)
    not_work_j = torch.tensor(not_work_j, dtype=torch.float)
    PT_list = torch.tensor(PT_list, dtype=torch.float)
    PT_tensor_list = []
    # for pts in PT_list:
    #     PT_tensor_list.append(torch.tensor(pts, dtype=torch.float))
    edge_index_M_list2 = []
    edge_index_M_PT_list2 = []
    for eds in edge_index_M_list:
        edge_index_M_list2.append(torch.tensor(eds,dtype=torch.long))

    for eds2 in edge_index_M_PT_list:
        edge_index_M_PT_list2.append(torch.tensor(eds2,dtype=torch.float))
    # print(PT_tensor_list)
    return x,mu,edge_index_M,edge_index_P,M_s_list,O_s_list,PT_list,M_s,o_dic_ind,edge_index_M_list2,not_work_j,PT_e,edge_index_M_PT_list2

if __name__ == '__main__':
    for index in range(5):
        data_set = []

        job_type_num = 5
        Machine_num=4
        Operation_num=[1,3]
        job_num=4
        dataset_size = 20000
        gamma = 0.5


        while(True):


            Mlist, Jlist, Olist, All_job_list = make_dataset(job_type_num = job_type_num,Machine_num=Machine_num,Operation_num=Operation_num,job_num=job_num)
            last_node = [i for i in range(len(Mlist))]
            a =schuduling(Mlist,All_job_list,Olist,option="Cmax",m1_CTime = 0,m2_CTime= 0,m3_Ctime = 0,m1_job1=None, m2_job2=None, m3_job3=None)
            # for i in All_job_list:
            #     print(len(i.get_Sequence()))
            XList,CDic,XStartDic,cmax = a.solves()
            c_max_list = [0 for i in range(Machine_num)]
            for i in range(Machine_num):
                cs_list = [0]
                for j in XList:
                    if int(j.split('_')[-1]) == i+1:
                        cs_list.append(CDic[j.split('_')[1]])
                c_max_list [i] = max(cs_list)


            XStartDic2 = copy.deepcopy(XStartDic)
            # print(XList)
            # print(CDic)
            XStartDic = sorted(XStartDic.items(), key = lambda item: item[1])
            done_node_list = []
            done_edge_list = []
            makespan_now = 0
            now_state_list = []
            now_next_list = []
            now_r_list = []
            now_y_list = []
            now_c_list = []
            now_action_list = []
            C_list = []
            fir =True
            for key, value in XStartDic:
                # print(str(key)[:-1]+"_"+str(key)[-1])
                x,mu,edge_index_M,edge_index_P,M_s_list,O_s_list,PT_list,M_s,o_dic_ind,edge_index_M_list,not_work_j,PT_e,edge_index_M_PT_list2 = makeinputdata(All_job_list, Mlist,
                                                                                               done_node_list, done_edge_list)
                if fir:
                    C_list = [[0] for i in range(len(x))]

                    fir = False

                state_now = [x, mu, edge_index_M, edge_index_P, M_s_list, O_s_list, PT_list,edge_index_M_list,not_work_j,PT_e,last_node,torch.tensor(C_list, dtype=torch.float),edge_index_M_PT_list2]
                print(last_node)
                action = []
                done_node_list.append(str(key))
                action.append(o_dic_ind[str(key)])
                # action.append(str(key))
                for e in XList:
                    if e.split("_")[1] == str(key):
                        done_edge_list.append(str(e)[2:])
                        if o_dic_ind[e.split("_")[2]][0] == "0":
                            first = "0" + e.split("_")[3]
                        else:
                            first = e.split("_")[2]

                        last_node[int(e.split("_")[3])-1] = int(o_dic_ind[e.split("_")[1]])
                        action.append([o_dic_ind[first],o_dic_ind[e.split("_")[1]]])
                        action.append(str(e)[2:])
                        # print(str(e)[2:])



                print(C_list)
                o_index = o_dic_ind[str(key)]
                C_list[int(o_index)] = [CDic[key]]

                print(C_list)

                if makespan_now < CDic[key]:
                    r = CDic[key] - makespan_now
                    makespan_now = CDic[key]
                else:
                    r = 0

                x,mu,edge_index_M,edge_index_P,M_s_list,O_s_list,PT_list,M_s,o_dic_ind,edge_index_M_list,not_work_j,PT_e,edge_index_M_PT_list2 = makeinputdata(All_job_list,Mlist,done_node_list,done_edge_list)
                y = makespan_now

                # print(edge_index_M_list)
                new_data = [x,mu,edge_index_M,edge_index_P,M_s_list,O_s_list,PT_list,edge_index_M_list,not_work_j,PT_e,last_node,torch.tensor(C_list, dtype=torch.float),edge_index_M_PT_list2]
                print(PT_e)
                now_state_list.append(state_now)
                now_next_list.append(new_data)
                now_r_list.append(r)
                now_y_list.append(y)
                now_c_list.append(CDic[key])
                now_action_list.append(action)

            print(edge_index_M_PT_list2)
            for i in range(len(now_state_list)):
                if i == 0:
                    target_y = 0

                else:
                    target_y = now_y_list[i-1]
                ind_t = 0
                target_c = c_max_list[int(now_action_list[i][-1].split('_')[-1])-1] - now_c_list[i]


                for j in range(i,len(now_r_list)):
                    target_y += now_r_list[j]*(gamma**ind_t)
                    target_c += now_r_list[j]*(gamma**ind_t)
                    # print(j)
                    # print(now_r_list[j])
                    # print(gamma**ind_t)
                    ind_t+=1

                print(target_y ," target!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(now_y_list[i])
                print(now_r_list[i])


                SARS = [now_state_list[i],now_action_list[i],now_next_list[i],now_r_list[i],now_y_list[i],target_y,cmax - now_c_list[i]]
                data_set.append(SARS)

            if len(data_set) > dataset_size:
                break

        #
        with open("datafinal1119_optimal_SARS_j{}_o{}_m{}_jt{}_gamma{}_ds{}_{}.pickle".format(job_num,Operation_num,Machine_num,job_type_num,gamma,dataset_size,index),"wb") as fw:
            pickle.dump(data_set,fw)




    # for graph in data_lst[start_idx:]:
    #     g_i += 1
    #     M = int(graph['M'][0].item())
    #     R = int(graph['R'][0].item())
    #     J = len(graph['x'])-(M+R)
    #     nx_graph = to_networkx(graph)
    #     colors = ['dodgerblue']*M + ['mediumorchid']*R + ['limegreen']*J
    #     plt.figure(g_i, figsize=(14, 12))
    #     nx.draw(nx_graph, cmap=plt.get_cmap('Set1'), node_color=colors, node_size=75, linewidths=6)
    #     if g_i == num_of_graph_figure:
    #         break
    # plt.show()




#graph = Data(x=x, edge_index=edge_index.t(), M_s=Ms, R_s=Rs, R_r=Rr, S_p=Sp, S_wt=Swt, S_r=Sr, S_a=Sa,
 #                    mu=mu, M=M, R=R)


