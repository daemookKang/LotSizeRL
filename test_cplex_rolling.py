import simpy
import numpy as np
import random
import copy
from dqn_setup_pro import DQN_FS
from nn import NN_FABSched
import matplotlib.pyplot as plt
import torch


class Job:
    def __init__(self, env=None, id: str='j1',processing_time={},demand_time=10,available_M_List=[]):
        self.env = env
        self.id = id
        self.process_time = processing_time
        self.demand_time = demand_time
        self.available_M_List = available_M_List
        self.completed_time = 0
        self.selected = False


    def __str__(self):
        return f'id: {self.id}\navailable_M_List: {self.available_M_List}\nprocess_time: {self.process_time}\ndemand_time: {self.demand_time}\n'


    def get_processing_time(self,mac):
        return self.process_time[mac]

    def set_fab_env(self,fab_env):
        self.fab_env = fab_env

    def process(self):
        chk =1
        while True:

            if self.env.now > chk*1440 :
                # print(int(self.id[1]))
                print(self.id)
                random.seed(chk + int(self.id[1]))
                print("now deamnd :; ",self.demand_time)
                self.demand_time = random.randint(4,4)
                print("after deamnd :; ", self.demand_time)
                print(self.env.now, " :: now")
                chk += 1
            if self.env.now-1 == self.fab_env.run_time:
                return 0
            self.fab_env.inventory_dic[self.id] -= 1
            self.fab_env.total_demand[self.id] -= 1

            if self.fab_env.inventory_dic[self.id] < 0:
                self.fab_env.inventory_dic[self.id] = 0
                self.fab_env.back_order_dic[self.id] += 1
                yield self.env.timeout(1)
            else:
                # self.fab_env.back_order_dic[self.id] = 0
                self.fab_env.total_make += 1
                yield self.env.timeout(self.demand_time)
                #count back-order
            # print(self.fab_env.inventory_dic , " :: inventory")
            # print(self.env.now, " :: now")



class Machine:
    def __init__(self, env=None, name: str = 'M1', init_setup_status=None,valid=False,action_list=None,lot_lst=None):
        self.env = env
        self.name = name
        self.action_lst = action_list
        self.lot_lst = lot_lst
        self.waiting_operation = []
        self.init_setup_status = init_setup_status
        self.state_lst = []
        self.action_path = []
        self.idle_time_lst = []
        self.first = True
        self.valid = valid
        self.now_idle = False
        self.occupied = False
        self.total_setup = 10000000000000000000

        self.last_action = True  # 마지막 action이 idle이었으면 False



    def __str__(self):
        return f'name: {self.name}\nsetup_status: {self.setup_status}\n'

    def process(self):
        index = -1
        while True:
            if self.env.now-1 == self.fab_env.run_time:
                return 0

            index += 1
            # if self.total_setup > 0:
            #     action_index,action_id = self.get_action()
            action_index = self.action_lst[index]
            action_id = self.fab_env.index_job_dic[action_index]


            # print(action_id , " :: action")
            print(self.init_setup_status , " -- > " ,action_id,"  :: ",self.name)
            print(self.fab_env.inventory_dic)
            if action_index == self.fab_env.job_index_dic[self.init_setup_status]:

                time_lst = []

                for i in range(self.lot_lst[index]):
                    yield self.env.timeout(self.fab_env.job_dict[self.init_setup_status].process_time[self.name])
                    self.fab_env.inventory_dic[self.init_setup_status] += 1
                # print(self.fab_env.inventory_dic, " :: action")

                time_lst.append(1)
                time_lst.append(0)
                total_back_order = 0
                for job in self.fab_env.job_dict.values():
                    if self in job.available_M_List:
                        total_back_order += self.fab_env.back_order_dic[job.id]
                        self.fab_env.back_order_dic[job.id] = 0

                time_lst.append(total_back_order)
                self.fab_env.total_cost += total_back_order

                state = []
                # state.extend(self.setup_status)
                state.extend(self.fab_env.now_set_up)
                can_m = []

                demand_now = []
                max_d = 0
                inventory_now = []
                max_i = 0

                for i in range(len(self.fab_env.can_machine_lst)):
                    if self in self.fab_env.job_dict[self.fab_env.index_job_dic[i]].available_M_List:
                        can_m.append(self.fab_env.can_machine_lst[i] - 1)
                    else:
                        can_m.append(self.fab_env.can_machine_lst[i])

                # state.extend(can_m)
                process_t = []
                max_p = 0

                for job in self.fab_env.job_dict.values():

                    if self.name in job.process_time.keys():
                        demand_now.append(job.demand_time)
                        process_t.append(job.process_time[self.name])
                        inventory_now.append(self.fab_env.inventory_dic[job.id] / job.demand_time)
                        if max_p < job.process_time[self.name]:
                            max_p = job.process_time[self.name]
                        if max_d < job.demand_time:
                            max_d = job.demand_time
                        if max_i < self.fab_env.inventory_dic[job.id] / job.demand_time:
                            max_i = self.fab_env.inventory_dic[job.id] / job.demand_time
                    else:
                        demand_now.append(0)
                        process_t.append(0)
                        inventory_now.append(-1)

                for job in range(len(self.fab_env.job_dict.values())):
                    demand_now[job] = demand_now[job] / 10
                    process_t[job] = process_t[job] / max_p
                    if inventory_now[job] == -1:
                        inventory_now[job] = 1
                    else:
                        inventory_now[job] = inventory_now[job] / max_i

                state.extend(demand_now)
                state.extend(inventory_now)
                state.extend(process_t)
                # state.extend([self.total_setup/4])
                self.state_lst.append(state)
                action_space = self.get_action_space()
                if self.total_setup > 0:
                    next_action = self.fab_env.dqn.get_greedy_action(self.state_lst[-1],action_space)
                else:
                    next_action = action_index
                if self.valid == False:
                    self.fab_env.dqn.train(self.state_lst, action_index, next_action, time_lst)


            else:
                time_lst = []
                self.total_setup -= 1
                key = self.init_setup_status + "_" + action_id
                time_lst.append(self.fab_env.job_dict[self.init_setup_status].process_time[self.name])
                self.fab_env.now_set_up[self.fab_env.job_index_dic[self.init_setup_status]] -= 1
                self.fab_env.now_set_up[self.fab_env.job_index_dic[action_id]] += 1
                self.init_setup_status = action_id
                yield self.env.timeout(self.fab_env.set_up_time_dic[key])

                for i in range(self.lot_lst[index]):


                    yield self.env.timeout(self.fab_env.job_dict[self.init_setup_status].process_time[self.name])
                    self.fab_env.inventory_dic[self.init_setup_status] += 1


                time_lst.append(self.fab_env.set_up_time_dic[key])
                total_back_order = 0
                for job in self.fab_env.job_dict.values():
                    if self in job.available_M_List:
                        total_back_order+=self.fab_env.back_order_dic[job.id]
                        self.fab_env.back_order_dic[job.id] = 0

                time_lst.append(total_back_order)
                # self.fab_env.total_cost += time_lst[1]/time_lst[0]
                self.fab_env.setup_chage_num += 1
                self.fab_env.total_cost+= total_back_order
                state = []
                # state.extend(self.setup_status)
                state.extend(self.fab_env.now_set_up)
                can_m = []

                demand_now = []
                max_d = 0
                inventory_now = []
                max_i = 0

                for i in range(len(self.fab_env.can_machine_lst)):
                    if self in self.fab_env.job_dict[self.fab_env.index_job_dic[i]].available_M_List:
                        can_m.append(self.fab_env.can_machine_lst[i] - 1)
                    else:
                        can_m.append(self.fab_env.can_machine_lst[i])

                # state.extend(can_m)
                process_t = []
                max_p = 0

                for job in self.fab_env.job_dict.values():

                    if self.name in job.process_time.keys():
                        demand_now.append(job.demand_time)
                        process_t.append(job.process_time[self.name])
                        inventory_now.append(self.fab_env.inventory_dic[job.id] / job.demand_time)
                        if max_p < job.process_time[self.name]:
                            max_p = job.process_time[self.name]
                        if max_d < job.demand_time:
                            max_d = job.demand_time
                        if max_i < self.fab_env.inventory_dic[job.id] / job.demand_time:
                            max_i = self.fab_env.inventory_dic[job.id] / job.demand_time
                    else:
                        demand_now.append(0)
                        process_t.append(0)
                        inventory_now.append(-1)

                for job in range(len(self.fab_env.job_dict.values())):
                    demand_now[job] = demand_now[job] / 10
                    process_t[job] = process_t[job] / max_p
                    if inventory_now[job] == -1:
                        inventory_now[job] = 1
                    else:
                        inventory_now[job] = inventory_now[job] / max_i

                state.extend(demand_now)
                state.extend(inventory_now)
                state.extend(process_t)
                # state.extend([self.total_setup/4])

                self.state_lst.append(state)
                action_space = self.get_action_space()
                if self.total_setup > 0:
                    next_action = self.fab_env.dqn.get_greedy_action(self.state_lst[-1], action_space)
                else:
                    next_action = action_index
                if self.valid == False:
                    self.fab_env.dqn.train(self.state_lst,action_index,next_action,time_lst)



    def set_init(self,fab_env,valid):
        self.fab_env = fab_env
        self.setup_status = []
        can_setup = []
        state = []
        can_job_list = []
        for job in self.fab_env.job_dict.values():
            self.setup_status.append(0)
            can_setup.append(0)

        for job in self.fab_env.job_dict.values():
            for m in job.available_M_List:
                if m.name == self.name:
                    can_job_list.append(job)
                    can_setup[self.fab_env.job_index_dic[job.id]] = 1

        if self.init_setup_status == None:
            if valid == True:
                random.seed(1)
            init_setup = random.choice(can_job_list)
            print("init :; ",init_setup)
            print("machine :: ",self.name)
            self.fab_env.now_set_up[self.fab_env.job_index_dic[init_setup.id]] += 1
            self.setup_status[self.fab_env.job_index_dic[init_setup.id]] = 1
            self.init_setup_status = init_setup.id
            # state.extend(self.setup_status)
            state.extend(self.fab_env.now_set_up)
            can_m = []

            demand_now = []
            max_d = 0
            inventory_now = []
            max_i = 0

            for i in range(len(self.fab_env.can_machine_lst)):
                if self in self.fab_env.job_dict[self.fab_env.index_job_dic[i]].available_M_List:
                    can_m.append(self.fab_env.can_machine_lst[i] - 1)
                else:
                    can_m.append(self.fab_env.can_machine_lst[i])

            # state.extend(can_m)
            process_t = []
            max_p = 0

            for job in self.fab_env.job_dict.values():

                if self.name in job.process_time.keys():
                    demand_now.append(job.demand_time)
                    process_t.append(job.process_time[self.name])
                    inventory_now.append(self.fab_env.inventory_dic[job.id] / job.demand_time)
                    if max_p < job.process_time[self.name]:
                        max_p = job.process_time[self.name]
                    if max_d < job.demand_time:
                        max_d = job.demand_time
                    if max_i < self.fab_env.inventory_dic[job.id] / job.demand_time:
                        max_i = self.fab_env.inventory_dic[job.id] / job.demand_time
                else:
                    demand_now.append(0)
                    process_t.append(0)
                    inventory_now.append(-1)

            for job in range(len(self.fab_env.job_dict.values())):
                demand_now[job] = demand_now[job] / 10
                process_t[job] = process_t[job] / max_p
                if inventory_now[job] == -1:
                    inventory_now[job] = 1
                else:
                    inventory_now[job] = inventory_now[job] / max_i

            state.extend(demand_now)
            state.extend(inventory_now)
            state.extend(process_t)
            # state.extend([self.total_setup/4])


            self.state_lst.append(state)
        else:
            self.setup_status[self.fab_env.job_index_dic[self.init_setup_status]] = 1
            self.fab_env.now_set_up[self.fab_env.job_index_dic[self.init_setup_status]] += 1


    def get_action_space(self):
        action_space = [0 for i in range(self.fab_env.num_job)]
        for job in self.fab_env.job_dict.values():
            for m in job.available_M_List:
                if m.name == self.name:

                    action_space[self.fab_env.job_index_dic[job.id]] = 1

        return action_space

    def get_action(self):
        if self.fab_env.env_type == 'dqn':
            state = []
            # state.extend(self.setup_status)
            state.extend(self.fab_env.now_set_up)
            can_m = []

            demand_now = []
            max_d = 0
            inventory_now = []
            max_i = 0

            for i in range(len(self.fab_env.can_machine_lst)):
                if self in self.fab_env.job_dict[self.fab_env.index_job_dic[i]].available_M_List:
                    can_m.append(self.fab_env.can_machine_lst[i] - 1)
                else:
                    can_m.append(self.fab_env.can_machine_lst[i])

            # state.extend(can_m)
            process_t = []
            max_p = 0

            for job in self.fab_env.job_dict.values():


                if self.name in job.process_time.keys():
                    demand_now.append(job.demand_time)
                    process_t.append(job.process_time[self.name])
                    inventory_now.append(self.fab_env.inventory_dic[job.id] / job.demand_time)
                    if max_p < job.process_time[self.name]:
                        max_p = job.process_time[self.name]
                    if max_d < job.demand_time:
                        max_d = job.demand_time
                    if max_i < self.fab_env.inventory_dic[job.id] / job.demand_time:
                        max_i = self.fab_env.inventory_dic[job.id] / job.demand_time
                else:
                    demand_now.append(0)
                    process_t.append(0)
                    inventory_now.append(-1)



            for job in range(len(self.fab_env.job_dict.values())):
                demand_now[job] = demand_now[job] / 10
                process_t[job] = process_t[job] / max_p
                if inventory_now[job] == -1:
                    inventory_now[job] = 1
                else:
                    inventory_now[job] = inventory_now[job] / max_i

            state.extend(demand_now)
            state.extend(inventory_now)
            state.extend(process_t)
            # state.extend([self.total_setup/4])


            self.state_lst.append(state)
            action_space = self.get_action_space()


            if self.valid == False:
                action = self.fab_env.dqn.get_eps_action(self.state_lst[-1],action_space)
            else:
                action = self.fab_env.dqn.get_greedy_action(self.state_lst[-1], action_space)
            self.setup_status = []
            for job in self.fab_env.job_dict.values():
                self.setup_status.append(0)

            self.setup_status[action] = 1

            return action, self.fab_env.index_job_dic[action]
        else:
            for j in range(len(self.fab_env.inventory_dic)):
                inv = self.fab_env.inventory_dic[self.fab_env.index_job_dic[j]]
                job = self.fab_env.job_dict[self.fab_env.index_job_dic[j]]
                if self in job.available_M_List:
                    demand = job.demand_time
                    remaing_t = 10/demand

                    if inv <= remaing_t + 3:
                        action = j

                        return action, self.fab_env.index_job_dic[action]
            return self.fab_env.job_index_dic[self.init_setup_status], self.init_setup_status




def scenario_generator(env, machine_num: int = 35,num_job=4,seed = 0,valid=False,num=0) -> (
dict, dict, dict, dict, dict):
    # Scenario parameters
    random.seed(2)
    min_process_time, max_process_time = (2, 3)
    min_setup_time, max_setup_time = (8, 10)
    min_demand_time, max_demand_time = (8, 10)
    min_inventory,max_inventory = (20,50)
    back_order_dic = {}

    # sqn_lst = [[1, 2, 0, 3, 1, 2, 0, 3], [1, 2, 3, 0, 3, 0, 3, 1, 2, 3], [1, 3, 2, 0, 1, 3, 1], [1, 3, 2, 0, 1, 3, 2],
    #            [1, 2, 3, 0, 1, 3, 2, 1, 2], [1, 3, 2, 0, 1, 3, 0, 3], [1, 3, 2, 0, 1, 3, 2, 3],
    #            [1, 0, 2, 3, 1, 0, 2, 0], [1, 3, 2, 0, 3, 1, 3, 1, 2], [1, 2, 0, 3, 0, 1, 2, 0], [1, 3, 2, 1, 0, 3, 1],
    #            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 2, 1, 2, 1, 2, 1, 3, 0, 2, 3, 0,
    #             1, 2, 3], [1, 2, 0, 3, 2, 1, 2, 1, 0, 2, 3], [1, 0, 2, 3, 1, 0, 2], [1, 3, 0, 1, 2, 3, 0, 1],
    #            [1, 3, 2, 0, 2, 0, 1, 3, 2], [1, 0, 3, 1, 2, 0, 3], [1, 3, 2, 0, 3, 1, 3, 1, 2, 3],
    #            [1, 3, 0, 2, 3, 1, 0]]
    # lot_lst = [[230, 225, 90, 60, 50, 20, 8, 2],
    #            [304, 151, 85, 4, 1, 64, 41, 25, 7, 3],
    #            [310, 185, 85, 65, 25, 15, 5], [220, 235, 125, 65, 25, 15, 5],
    #            [245, 180, 115, 85, 25, 25, 5, 2, 3],
    #            [170, 235, 115, 80, 40, 25, 15, 5],
    #            [173, 242, 105, 76, 44, 25, 12, 11],
    #            [180, 205, 145, 70, 40, 25, 15, 5],
    #            [205, 210, 145, 75, 34, 4, 1, 8, 5],
    #            [185, 250, 85, 75, 46, 24, 15, 5],
    #            [200, 235, 105, 75, 35, 25, 10],
    #            [180, 5, 8, 10, 8,
    #             10, 8, 10, 8, 10,
    #             8, 10, 125, 2, 10, 15,
    #             5, 10, 10, 10,
    #             1, 10, 5, 43, 43,
    #             2, 10, 5, 10, 35,
    #             15, 4, 2, 1, 11,
    #             3, 1, 3],
    #            [240, 145, 115, 80, 40, 4, 1, 28, 15, 6, 4],
    #            [270, 210, 110, 45, 35, 15, 5],
    #            [250, 155, 125, 75, 40, 20, 15, 5],
    #            [314, 186, 79, 4, 1, 48, 35, 15, 4],
    #            [190, 225, 145, 75, 35, 15, 5],
    #            [242, 153, 145, 67, 33, 4, 1, 20, 15, 4],
    #            [190, 225, 145, 75, 35, 14, 5]]

    sqn_lst = [[1, 3, 0, 2, 1, 3, 1], [1, 3, 2, 0, 3, 2, 1, 2], [1, 0, 3, 1, 0, 2, 3, 1], [1, 3, 0, 1, 2, 0, 3],
               [1, 0, 2, 3, 0, 2, 1, 2], [1, 3, 0, 1, 2, 3, 1, 2], [1, 3, 0, 2, 1, 0, 1, 0, 3, 0],
               [1, 3, 0, 2, 1, 2, 1, 3, 0, 3], [1, 2, 3, 0, 2, 1, 0, 2], [1, 0, 3, 2, 0, 1, 3],
               [1, 0, 2, 1, 3, 2, 0, 2], [1, 3, 2, 0, 1, 3, 0, 2], [1, 3, 0, 2, 1, 3, 0, 3, 0, 2, 0, 2],
               [1, 2, 0, 2, 0, 3, 1, 3, 1, 2, 0, 2, 0, 1, 3], [1, 2, 0, 3, 1, 2, 0], [1, 0, 2, 3, 1, 0, 1, 2],
               [1, 3, 2, 0, 1, 3, 2], [1, 0, 3, 2, 0, 3, 0], [1, 3, 2, 0, 3, 1, 3, 1, 2, 3], [1, 2, 0, 3, 2, 1, 0]]
    lot_lst = [[245.0, 205.0, 104.0, 61.0, 34.99999999999999, 25.0, 10.000000000000007],
               [251.0, 134.0, 135.0, 76.0, 44.00000000000001, 25.0, 15.0, 5.0],
               [240.0, 135.0, 155.0, 85.0, 35.0, 15.0, 15.0, 5.0],
               [285.0, 210.0, 95.0, 57.0, 23.0, 14.999999999999996, 5.0],
               [220.0, 215.0, 125.0, 75.0, 35.0, 14.0, 4.000000000000001, 0.9999999999999991],
               [240.0, 215.0, 130.0, 31.0, 34.0, 25.0, 8.0, 1.9999999999999956],
               [170.0, 265.0, 116.0, 64.0, 35.0, 4.0, 1.0, 15.0, 10.000000000000002, 5.0],
               [175.0, 240.0, 108.0, 72.0, 3.0, 2.0, 40.0, 25.0, 12.0, 4.999999999999999],
               [205.0, 200.0, 115.0, 75.0, 50.0, 20.0, 15.0, 5.0],
               [205.0, 220.0, 135.0, 75.00000000000001, 35.0, 12.0, 5.0],
               [160.0, 265.0, 115.0, 60.0, 40.0, 25.0, 14.999999999999996, 5.0],
               [230.0, 210.0, 90.0, 65.0, 55.0, 25.0, 7.0, 2.9999999999999916],
               [165.0, 180.0, 128.0, 82.0, 55.0, 35.0, 4.0, 1.0, 12.0, 10.0, 4.0, 0.9999999999999982],
               [230.0, 165.0, 2.0, 3.0, 110.0, 74.0, 3.0, 2.0, 40.0, 25.0, 1.9999999999999654, 3.0000000000000346,
                12.000000000000036, 3.9999999999999227, 4.0], [273.0, 202.0, 115.0, 45.0, 35.0, 15.0, 5.0],
               [300.0, 165.0, 95.0, 60.0, 30.0, 25.0, 7.0, 3.0], [310.0, 190.0, 80.0, 52.0, 35.0, 15.0, 4.0],
               [330.0, 165.0, 95.0, 55.0, 25.000000000000004, 15.0, 5.0],
               [243.0, 152.0, 145.0, 68.0, 32.0, 4.0, 1.0, 20.0, 15.000000000000002, 4.0],
               [190.0, 225.0, 145.0, 75.0, 35.0, 11.999999999999996, 5.0]]

    # sqn_lst = [[1, 3, 0, 2, 1, 2, 1, 3, 0, 3], [1, 0, 2, 3, 1, 3, 1, 0, 1, 0, 2], [1, 3, 0, 2, 3, 0, 3, 0, 1],
    #            [1, 0, 3, 2, 3, 2, 0, 2, 0, 1], [1, 3, 0, 2, 3, 1, 2], [1, 0, 1, 0, 3, 1, 0, 1, 0, 3, 2, 3],
    #            [1, 0, 3, 0, 3, 0, 2, 1, 2, 1, 3], [1, 3, 2, 3, 2, 0, 2, 0, 3, 1, 3, 1, 2], [1, 2, 3, 0, 1, 2, 0, 2],
    #            [1, 0, 1, 0, 2, 3, 0, 3, 0, 1, 2], [1, 0, 1, 2, 3, 0, 3, 0, 3, 1, 3, 1, 2],
    #            [1, 2, 0, 2, 0, 3, 1, 3, 1, 2, 3], [1, 3, 0, 2, 1, 2, 1, 3, 0, 3, 0, 2], [1, 2, 0, 3, 1, 2, 0, 2, 0, 3],
    #            [1, 2, 0, 2, 0, 3, 1, 3, 1, 2], [1, 0, 3, 2, 3, 2, 1, 2, 0, 1, 0, 1], [1, 0, 1, 0, 3, 1, 2, 3, 1, 3],
    #            [1, 2, 0, 2, 0, 3, 2, 3, 0, 2], [1, 2, 0, 2, 0, 3, 1, 2, 0, 1, 2], [1, 3, 2, 0, 2, 0, 2, 3, 2, 3]]
    # lot_lst = [[240.0, 205.0, 68.0, 72.0, 13.0, 12.0, 40.0, 15.0, 12.0, 12.999999999999991],
    #            [240.0, 123.0, 167.0, 55.00000000000001, 9.0, 16.0, 30.0, 12.0, 13.0, 12.000000000000004,
    #             12.999999999999996], [315.0, 160.0, 68.0, 63.0, 34.0, 14.0, 11.0, 10.0, 15.0],
    #            [325.0, 160.0, 105.0, 10.0, 15.0, 30.0, 19.99999999999999, 5.000000000000011, 10.00000000000001, 15.0],
    #            [240.0, 170.0, 150.0, 65.0, 45.0, 10.0, 10.0],
    #            [210.0, 22.000000000000014, 2.999999999999986, 190.0, 135.0, 55.0, 20.0, 5.0, 22.99999999999999,
    #             2.0000000000000107, 12.000000000000014, 12.99999999999999],
    #            [180.0, 155.00000000000003, 165.0, 10.000000000000014, 14.999999999999986, 90.0, 25.0,
    #             14.000000000000004, 10.999999999999996, 9.999999999999996, 15.000000000000004],
    #            [196.00000000000017, 188.99999999999983, 8.000000000000014, 16.999999999999986, 90.0, 12.0, 13.0, 72.0,
    #             43.000000000000014, 9.999999999999828, 15.000000000000172, 12.000000000000004, 11.999999999999831],
    #            [188.0, 240.0, 102.0, 76.0, 40.0, 19.0, 14.0, 11.0],
    #            [180.0, 15.0, 10.0, 190.0, 140.00000000000003, 80.0, 15.0, 10.0, 30.0, 4.9999999999999964,
    #             15.000000000000004],
    #            [150.0, 7.105427357601002e-13, 53.0, 241.9999999999993, 55.00000000000001, 15.0, 10.0, 90.0,
    #             25.000000000000004, 14.000000000000004, 10.999999999999996, 9.999999999999996, 15.000000000000004],
    #            [240.0, 145.0, 12.0, 13.0, 120.0, 55.0, 3.0, 22.0, 45.0, 22.0, 13.0],
    #            [160.0, 195.0, 131.0, 68.99999999999999, 13.999999999999993, 11.000000000000007, 46.00000000000001, 9.0,
    #             1.000000000000012, 24.0, 12.000000000000004, 12.000000000000004],
    #            [217.0, 168.0, 85.99999999999976, 84.00000000000024, 60.0, 20.0, 12.0, 12.999999999999684, 12.0,
    #             11.999999999999757],
    #            [279.0, 196.0, 11.000000000000028, 13.999999999999972, 105.0, 40.000000000000014, 19.000000000000007,
    #             5.999999999999993, 10.0, 15.0],
    #            [300.0, 190.0, 70.0, 12.0, 13.0, 30.0, 7.0, 18.0, 10.0, 15.0, 9.999999999999995, 15.00000000000001],
    #            [240.0, 12.0, 13.0, 180.0, 80.0, 72.0, 48.0, 20.0, 15.0, 10.0],
    #            [330.0, 145.0, 15.0, 10.0, 90.0, 55.0, 20.000000000000004, 4.9999999999999964, 15.000000000000004,
    #             9.999999999999996],
    #            [231.0, 214.0, 21.999999999999517, 3.000000000000483, 74.00000000000048, 70.99999999999952,
    #             35.9999999999515, 14.000000000048505, 12.0, 4.850519985666324e-11, 12.999999999951015],
    #            [210.0, 235.0, 85.0, 15.0, 10.0, 75.0, 24.999999999999996, 10.000000000000004, 15.000000000000004,
    #             9.999999999999996]]

    machine_dict = {}

    action_lst = sqn_lst[num]
    lot_lst = lot_lst[num]
    lot_lst2 = []
    for lot in lot_lst:
        lot_lst2.append(int(lot))
    lot_lst2[-1] = 10000

    # print(sum(lot_lst)*2)
    # print(len(action_lst))
    # print(len(lot_lst))

    for m in range(machine_num):
        machine_name = f'M{m + 1}'
        machine_dict[machine_name] = Machine(env, machine_name,None,valid,action_lst,lot_lst2)


    inventory_dic = {}

    job_dict = {}
    job_idx = 1
    type_inx = 0
    set_up_time_dic = {}

    for job_type in range(num_job):
        available_M_num = random.randint(1, len(machine_dict.values()))
        am_list = list(machine_dict.values())

        available_M_List = random.sample(am_list, available_M_num)
        processing_time = {}


        for machine in available_M_List:
            processing_time2 = random.randint(min_process_time, max_process_time)
            processing_time2 = 2
            processing_time[machine.name] = processing_time2

        demand_time = random.randint(min_demand_time, max_demand_time)



        job_id = f'j{job_idx}'
        job = Job(env,job_id,processing_time,demand_time,available_M_List)
        job_dict[job_id] = job

        back_order_dic[job_id] = 0
        job_idx += 1
        type_inx += 1

    for job1 in job_dict.values():
        for job2 in job_dict.values():
            # print(job1)
            # print(job2)
            key = job1.id+"_"+job2.id

            if job1.id == job2.id:
                set_up_time2 = 0
            else:
                set_up_time2 = random.randint(min_setup_time, max_setup_time)
                set_up_time2 = 10
            set_up_time_dic[key] = set_up_time2

    random.seed(seed+10)
    for job in job_dict.values():
        inventory_dic.update({job.id:random.randint(min_inventory, max_inventory)})




    return machine_dict,  job_dict, set_up_time_dic, back_order_dic


class FABEnvironment:
    def __init__(self, env_type: str = 'dqn', dqn_agent=None,run_time=5000):
        self.sim_env = simpy.Environment()
        self.env_type = env_type
        self.back_order_dic = {}
        self.results = []
        self.completed_job_lst = list()
        self.machine_num = 1
        self.total_cost = 0
        self.total_make = 0

        self.run_time=run_time
        self.num_job = 4

        self.min_job_num_for_a_job_type = 20
        self.max_job_num_for_a_job_type = 24

        self.dqn = dqn_agent
        self.valid_lst = []
        self.valid_make = []
        self.valid_setup_lst = []
        self.seed_lst = [seed for seed in range(500)]

    def set_scenario(self,seed,valid,init_inv=None):

        self.now_remaing_t = []
        self.can_machine_lst = []
        self.now_set_up = []
        self.inventory_dic = {}
        self.total_demand = {}
        self.total_cost = 0
        self.setup_chage_num = 0
        self.total_make = 0
        min_demand_time, max_demand_time = (4, 5)
        self.machine_dict,  self.job_dict , self.set_up_time_dic,self.back_order_dic = scenario_generator(self.sim_env, self.machine_num, self.num_job,seed = 1,valid=valid,num=seed)
        random.seed(seed + 100)
        min_inventory, max_inventory = (10, 80)
        for job in self.job_dict.values():
            self.inventory_dic.update({job.id: random.randint(min_inventory, max_inventory)})
            job.demand_time = random.randint(min_demand_time, max_demand_time)
            self.total_demand.update({job.id:int(2880/job.demand_time)})
        if init_inv != None and init_inv != 'test':
            self.inventory_dic = init_inv

        lstInitInv = [8,10,10,12]
        intn = 0
        if init_inv == 'test':
            for i in self.job_dict.values():

                self.inventory_dic[i.id] = lstInitInv[intn]
                intn +=1

        self.job_index_dic = {}
        self.index_job_dic = {}
        # print(self.inventory_dic," :: inv")
        # print(self.set_up_time_dic, " :: setup")
        #
        for j in self.job_dict.values():
            print(j.id, " :: job id")
            print(j.process_time, " :: demand")
            print(j.demand_time, " :: demand")

        index = 0

        for job in self.job_dict.values():
            self.job_index_dic.update({job.id: index})
            self.index_job_dic.update({index : job.id})
            self.can_machine_lst.append(len(job.available_M_List))
            self.now_set_up.append(0)


            index += 1


        # self.sim_env.process(self.machine_dict[f'M{m+1}'])

    # def reset_scenario(self):
    #     for job in self.job_dict.values():
    #         job.selected = False
    #         job.done = False
    #         job.cur_step_num = 0
    #         job.completed_time = 0
    #
    #     ind = 0
    #     type_idx = 0
    #     for job in self.job_type_dict.values():
    #         self.job_num_dic.update({job.name: self.job_num_for_a_job_type_lst[type_idx]})
    #         for oper in range(len(job.operation)):
    #             self.job_oper_index_dic.update({job.name +"_"+str(oper):ind})
    #
    #             ind +=1
    #         type_idx += 1
    #
    #     for m in self.machine_dict.values():
    #         m.state_lst = []
    #         m.action_path = []
    #         m.first = True
    #         m.last_action = True
    #
    #     self.completed_job_lst = []
    #     self.sim_env = simpy.Environment()

    def simulation_run(self,  max_scenario_num=20):
        if self.env_type == 'dqn':
            for scenario in range(max_scenario_num):
                seed =random.choice(self.seed_lst)
                if  scenario >= 0:
                    valid = True
                    self.set_scenario(scenario,True,'test')
                else:
                    valid = True
                    self.set_scenario(scenario,True,self.inventory_dic)

                for j in range(self.num_job):
                    self.job_dict[f'j{j+1}'].set_fab_env(self)
                    self.sim_env.process(self.job_dict[f'j{j+1}'].process())
                for m in range(self.machine_num):
                    self.machine_dict[f'M{m + 1}'].set_init(self,valid)
                    self.sim_env.process(self.machine_dict[f'M{m + 1}'].process())
                self.sim_env.run(until=self.run_time)
                self.sim_env = simpy.Environment()
                self.dqn.now_step +=1
                # if scenario%10 == 0 and scenario>0:
                #     self.dqn.update_dqn()

                if scenario >= 0:
                    self.valid_lst.append(self.total_cost)
                    self.valid_setup_lst.append(self.setup_chage_num)
                    print(self.results, " result")
                    print(self.valid_lst , " valid")
                    print(self.valid_setup_lst, " setup")
                    print("mean = ",np.mean(np.array(self.valid_lst)))
                else:
                    self.results.append(self.total_cost)
        if self.env_type == 'heu':
            for scenario in range(max_scenario_num):
                seed =random.choice(self.seed_lst)
                if  scenario >= 0:
                    valid = True
                    self.set_scenario(scenario,True)
                else:
                    valid = True
                    self.set_scenario(scenario,True,self.inventory_dic)

                for j in range(self.num_job):
                    self.job_dict[f'j{j+1}'].set_fab_env(self)
                    self.sim_env.process(self.job_dict[f'j{j+1}'].process())
                for m in range(self.machine_num):
                    self.machine_dict[f'M{m + 1}'].set_init(self,valid)
                    self.sim_env.process(self.machine_dict[f'M{m + 1}'].process())
                self.sim_env.run(until=self.run_time)
                self.sim_env = simpy.Environment()
                self.dqn.now_step +=1
                # if scenario%10 == 0 and scenario>0:
                #     self.dqn.update_dqn()

                if scenario >= 0:
                    self.valid_lst.append(self.total_cost)
                    self.valid_setup_lst.append(self.setup_chage_num)
                    self.valid_make.append(self.total_make)
                    print(self.results, " result")
                    print(self.valid_lst , " valid")
                    print(self.valid_setup_lst, " setup")
                    print("total make :; ", self.valid_make)

                    print("mean = ",np.mean(np.array(self.valid_lst)))

                    print("mean make :: ",np.mean(np.array(self.valid_make)))
                else:
                    self.results.append(self.total_cost)
                    print("total idle :: ", self.idle_time)




        else:
            pass

#
# class MachineSelector:
#     def __init__(self, env, fab_env):
#         self.env = env
#         self.fab_env = fab_env
#         self.eps = 0.01
#         # self.env.process(self.process())
#
#     def process(self):
#         while True:
#
#             end_condition = []
#             for job in self.fab_env.job_dict.values():
#                 end_condition = end_condition + [job.done]
#
#             if all(end_condition):
#                 return 0
#
#             self.avail_machine_lst = []
#             q_val_lst = []
#             max_q_val = -99999
#             # index_m = 0
#             index = 0
#             # print(11111111)
#
#             for machine in self.fab_env.machine_dict.values():
#                 if machine.occupied == False:
#                     self.avail_machine_lst.append(machine)
#             # print(22222222)
#             if bool(self.avail_machine_lst):
#                 # print("aa")
#                 for avail_m in self.avail_machine_lst:
#                     # print("hye")
#                     if len(avail_m.state_lst) > 0:
#                         now_state = avail_m.state_lst[-1]
#                     else:
#                         avail_m.set_state(self.fab_env, self.env)
#                         now_state = avail_m.state
#                     now_action_space = avail_m.get_action_space()
#                     q_v = self.fab_env.dqn.get_q_value(now_state,now_action_space)
#                     # print("bb")
#                     if max_q_val < q_v:
#                         max_q_val = q_v
#                         index_m  = index
#                     q_val_lst.append(self.fab_env.dqn.get_q_value(now_state,now_action_space))
#                     index+=1
#                     # print("ee")
#                 # print(q_val_lst)
#                 # print("cc")
#                 p = np.random.rand()
#                 if p < self.eps:
#                     chosen_machine = random.choice(self.avail_machine_lst)  # 머신 주기
#                     chosen_machine.occupied = True
#                 else:
#
#                     chosen_machine = self.avail_machine_lst[index_m]  # 머신 주기
#                     chosen_machine.occupied = True
#                 # chosen_machine = machine choosing algorithm
#                 # if chosen_machine:
#                 #   chosen_machine.chosen = True
#                 # else:
#                 #   yield self.env.timeout(1)
#                 # print("dd")
#             else:
#                 # print(4444444444)
#                 yield self.env.timeout(1)
#                 # print(self.env.now," now!!!!!!!!!!!!!!!!!!")
#
#                 # print(5555555555)
#             # print(33333333)

#
input_dim = 16
output_dim = 4
model = NN_FABSched(input_dim,output_dim)
# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4950epochs77max.pt"))
# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4950epochsall_setup.pt"))
# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4950epochslot5.pt"))
# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4950epochslot20.pt"))
# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4950epochslot30.pt"))
# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4950epochsall_setup_all.pt"))

# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4950epochssetup_16feat.pt"))

# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4950epochsmodel1_lot30.pt"))
# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4950epochssetup_process.pt"))
# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4950epochs7setup.pt"))
#
dqn = DQN_FS(model,learning_rate=10**(-3),env_name="test")

sqn_lst= [[1, 3, 0, 2, 1, 2, 1, 3, 0, 3], [1, 0, 2, 3, 1, 3, 1, 0, 1, 0, 2], [1, 3, 0, 2, 3, 0, 3, 0, 1], [1, 0, 3, 2, 3, 2, 0, 2, 0, 1], [1, 3, 0, 2, 3, 1, 2], [1, 0, 1, 0, 3, 1, 0, 1, 0, 3, 2, 3], [1, 0, 3, 0, 3, 0, 2, 1, 2, 1, 3], [1, 3, 2, 3, 2, 0, 2, 0, 3, 1, 3, 1, 2], [1, 2, 3, 0, 1, 2, 0, 2], [1, 0, 1, 0, 2, 3, 0, 3, 0, 1, 2], [1, 0, 1, 2, 3, 0, 3, 0, 3, 1, 3, 1, 2], [1, 2, 0, 2, 0, 3, 1, 3, 1, 2, 3], [1, 3, 0, 2, 1, 2, 1, 3, 0, 3, 0, 2], [1, 2, 0, 3, 1, 2, 0, 2, 0, 3], [1, 2, 0, 2, 0, 3, 1, 3, 1, 2], [1, 0, 3, 2, 3, 2, 1, 2, 0, 1, 0, 1], [1, 0, 1, 0, 3, 1, 2, 3, 1, 3], [1, 2, 0, 2, 0, 3, 2, 3, 0, 2], [1, 2, 0, 2, 0, 3, 1, 2, 0, 1, 2], [1, 3, 2, 0, 2, 0, 2, 3, 2, 3]]
lot_lst= [[240.0, 205.0, 68.0, 72.0, 13.0, 12.0, 40.0, 15.0, 12.0, 12.999999999999991], [240.0, 123.0, 167.0, 55.00000000000001, 9.0, 16.0, 30.0, 12.0, 13.0, 12.000000000000004, 12.999999999999996], [315.0, 160.0, 68.0, 63.0, 34.0, 14.0, 11.0, 10.0, 15.0], [325.0, 160.0, 105.0, 10.0, 15.0, 30.0, 19.99999999999999, 5.000000000000011, 10.00000000000001, 15.0], [240.0, 170.0, 150.0, 65.0, 45.0, 10.0, 10.0], [210.0, 22.000000000000014, 2.999999999999986, 190.0, 135.0, 55.0, 20.0, 5.0, 22.99999999999999, 2.0000000000000107, 12.000000000000014, 12.99999999999999], [180.0, 155.00000000000003, 165.0, 10.000000000000014, 14.999999999999986, 90.0, 25.0, 14.000000000000004, 10.999999999999996, 9.999999999999996, 15.000000000000004], [196.00000000000017, 188.99999999999983, 8.000000000000014, 16.999999999999986, 90.0, 12.0, 13.0, 72.0, 43.000000000000014, 9.999999999999828, 15.000000000000172, 12.000000000000004, 11.999999999999831], [188.0, 240.0, 102.0, 76.0, 40.0, 19.0, 14.0, 11.0], [180.0, 15.0, 10.0, 190.0, 140.00000000000003, 80.0, 15.0, 10.0, 30.0, 4.9999999999999964, 15.000000000000004], [150.0, 7.105427357601002e-13, 53.0, 241.9999999999993, 55.00000000000001, 15.0, 10.0, 90.0, 25.000000000000004, 14.000000000000004, 10.999999999999996, 9.999999999999996, 15.000000000000004], [240.0, 145.0, 12.0, 13.0, 120.0, 55.0, 3.0, 22.0, 45.0, 22.0, 13.0], [160.0, 195.0, 131.0, 68.99999999999999, 13.999999999999993, 11.000000000000007, 46.00000000000001, 9.0, 1.000000000000012, 24.0, 12.000000000000004, 12.000000000000004], [217.0, 168.0, 85.99999999999976, 84.00000000000024, 60.0, 20.0, 12.0, 12.999999999999684, 12.0, 11.999999999999757], [279.0, 196.0, 11.000000000000028, 13.999999999999972, 105.0, 40.000000000000014, 19.000000000000007, 5.999999999999993, 10.0, 15.0], [300.0, 190.0, 70.0, 12.0, 13.0, 30.0, 7.0, 18.0, 10.0, 15.0, 9.999999999999995, 15.00000000000001], [240.0, 12.0, 13.0, 180.0, 80.0, 72.0, 48.0, 20.0, 15.0, 10.0], [330.0, 145.0, 15.0, 10.0, 90.0, 55.0, 20.000000000000004, 4.9999999999999964, 15.000000000000004, 9.999999999999996], [231.0, 214.0, 21.999999999999517, 3.000000000000483, 74.00000000000048, 70.99999999999952, 35.9999999999515, 14.000000000048505, 12.0, 4.850519985666324e-11, 12.999999999951015], [210.0, 235.0, 85.0, 15.0, 10.0, 75.0, 24.999999999999996, 10.000000000000004, 15.000000000000004, 9.999999999999996]]

#
#
env = FABEnvironment(env_type='heu',dqn_agent=dqn,run_time=1440)
# # env.set_scenario()
env.simulation_run()


