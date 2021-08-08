import simpy
import numpy as np
import random
import copy
from dqn_cnn_cpu import DQN_FS
from cnn_low_deuling import NN_FABSched
import matplotlib.pyplot as plt
import time
import torch

class Job:
    def __init__(self, env=None, id: str = 'j1', processing_time={}, demand_time=10, available_M_List=[]):
        self.env = env
        self.id = id
        self.process_time = processing_time
        self.demand_time = demand_time
        self.available_M_List = available_M_List
        self.completed_time = 0
        self.selected = False

    def __str__(self):
        return f'id: {self.id}\navailable_M_List: {self.available_M_List}\nprocess_time: {self.process_time}\ndemand_time: {self.demand_time}\n'

    def get_processing_time(self, mac):
        return self.process_time[mac]

    def set_fab_env(self, fab_env):
        self.fab_env = fab_env

    def process(self):
        while True:
            if self.env.now - 1 == self.fab_env.run_time:
                return 0
            self.fab_env.inventory_dic[self.id] -= 1

            if self.fab_env.inventory_dic[self.id] < 0:
                self.fab_env.inventory_dic[self.id] = 0
                self.fab_env.back_order_dic[self.id] += 1
                self.fab_env.total_idle += 1
                yield self.env.timeout(1)
            else:
                self.fab_env.total_demand[self.id] -= 1
                self.fab_env.total_make += 1
                # self.fab_env.back_order_dic[self.id] = 0
                yield self.env.timeout(self.demand_time)
                # count back-order
            # print(self.fab_env.inventory_dic , " :: inventory")
            # print(self.env.now, " :: now")


class Machine:
    def __init__(self, env=None, name: str = 'M1', init_setup_status=None, valid=False):
        self.env = env
        self.name = name
        self.waiting_operation = []
        self.init_setup_status = init_setup_status
        self.state_lst = []
        self.action_path = []
        self.idle_time_lst = []
        self.first = True
        self.valid = valid
        self.now_idle = False
        self.occupied = False
        self.total_setup = 100000

        self.last_action = True  # 마지막 action이 idle이었으면 False

    def __str__(self):
        return f'name: {self.name}\nsetup_status: {self.setup_status}\n'


    def last_action_F(self,total_back_order,time_lst,action_index):
        total_back_order = total_back_order
        time_lst = time_lst
        action_index = action_index

        #time_lst.append(1)
        time_lst.append(self.env.now)
        time_lst.append(0)

        ramining_time = 0

        for job in self.fab_env.job_dict.values():
            if self in job.available_M_List:
                total_back_order += (self.fab_env.back_order_dic[job.id] / job.demand_time) / len(job.available_M_List)
                ramining_time += job.demand_time * self.fab_env.inventory_dic[job.id]

        time_lst.append(total_back_order)
        time_lst.append(ramining_time)
        time_lst.append(self.fab_env.job_dict[self.init_setup_status].demand_time)
        self.fab_env.total_cost += total_back_order

        state = []

        setup_me = []
        setups = []
        for i in self.fab_env.now_set_up:
            setups.append(i / self.fab_env.machine_num)

        for i in range(self.fab_env.num_job):
            key = self.init_setup_status + "_" + self.fab_env.index_job_dic[i]
            setup_me.append(self.fab_env.set_up_time_dic[key] / 20)
        state.append(setup_me)
        state.append(setups)
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
        demand_feat = []
        for job in self.fab_env.job_dict.values():

            if self.name in job.process_time.keys():
                demand_now.append(job.demand_time)
                process_t.append(job.process_time[self.name])
                inventory_now.append(self.fab_env.inventory_dic[job.id])
                demand_feat.append(
                    max(0, self.fab_env.total_demand[job.id] - self.fab_env.inventory_dic[job.id]) / 100)
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
                demand_feat.append(0)

        demand_process = []
        for job in range(len(self.fab_env.job_dict.values())):
            demand_now[job] = demand_now[job] / 10
            process_t[job] = process_t[job] / max_p
            # demand_process.append(demand_now[job]/process_t[job])
            if inventory_now[job] == -1:
                inventory_now[job] = 1
            else:
                inventory_now[job] = inventory_now[job] / 100

        state.append(demand_now)
        state.append(inventory_now)

        state.append(demand_feat)

        self.state_lst.append(state)
        action_space = self.get_action_space()

        next_action = 'no'

        if self.valid == False:
            self.fab_env.dqn.train(self.state_lst, action_index, next_action, time_lst)
        # print("last action!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        return "sss"

    def process(self):
        while True:
            if self.env.now == 960:
                print("END!!!!!!!!!!!!!!!!!!")
                return 0

            if self.total_setup > 0:
                action_index, action_id = self.get_action()
            # print(action_id , " :: action")
            if action_index == self.fab_env.job_index_dic[self.init_setup_status]:

                time_lst = []
                time_lst.append(self.env.now)
                time_lst.append(0)
                total_back_order = 0
                for job in self.fab_env.job_dict.values():
                    if self in job.available_M_List:
                        total_back_order -= (self.fab_env.back_order_dic[job.id] / job.demand_time) / len(
                            job.available_M_List)

                for i in range(self.fab_env.lot_size):
                    yield self.env.timeout(self.fab_env.job_dict[self.init_setup_status].process_time[self.name])
                    self.fab_env.inventory_dic[self.init_setup_status] += 1
                    # print(self.env.now)
                    if self.env.now + 2 == self.fab_env.run_time or self.env.now == self.fab_env.run_time:
                        self.last_action_F(total_back_order,time_lst,action_index)
                        return 0
                # print(self.fab_env.inventory_dic, " :: action")


                time_lst.append(self.env.now)


                ramining_time = 0
                for job in self.fab_env.job_dict.values():
                    if self in job.available_M_List:
                        total_back_order += (self.fab_env.back_order_dic[job.id] / job.demand_time) / len(
                            job.available_M_List)
                        ramining_time += job.demand_time * self.fab_env.inventory_dic[job.id]
                        # self.fab_env.back_order_dic[job.id] = 0

                time_lst.append(total_back_order)
                time_lst.append(ramining_time)
                time_lst.append(self.fab_env.job_dict[self.init_setup_status].demand_time)
                self.fab_env.total_cost += total_back_order

                state = []
                # state.extend(self.setup_status)
                setup_me = []
                setups = []
                for i in self.fab_env.now_set_up:
                    setups.append(i / self.fab_env.machine_num)
                # for i in range(self.fab_env.num_job):
                #     if i == self.fab_env.job_index_dic[self.init_setup_status]:
                #         setup_me.append(1)
                #     else:
                #         setup_me.append(0)
                for i in range(self.fab_env.num_job):
                    key = self.init_setup_status + "_" + self.fab_env.index_job_dic[i]
                    setup_me.append(self.fab_env.set_up_time_dic[key] / 20)
                state.append(setup_me)
                state.append(setups)
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
                demand_feat = []
                for job in self.fab_env.job_dict.values():

                    if self.name in job.process_time.keys():
                        demand_now.append(job.demand_time)
                        process_t.append(job.process_time[self.name])
                        inventory_now.append(self.fab_env.inventory_dic[job.id])
                        demand_feat.append(
                            max(0, self.fab_env.total_demand[job.id] - self.fab_env.inventory_dic[job.id]) / 100)
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
                        demand_feat.append(0)

                demand_process = []
                for job in range(len(self.fab_env.job_dict.values())):
                    demand_now[job] = demand_now[job] / 10
                    process_t[job] = process_t[job] / max_p
                    # demand_process.append(demand_now[job]/process_t[job])
                    if inventory_now[job] == -1:
                        inventory_now[job] = 1
                    else:
                        inventory_now[job] = inventory_now[job] / 100

                state.append(demand_now)
                # state.append(process_t)
                # state.extend(demand_process)
                state.append(inventory_now)

                # state.extend([self.total_setup/4])

                #         sum_demand = 0
                #         for k in self.fab_env.max_demand.keys():
                #             sum_demand += self.fab_env.total_demand[k]
                #         if sum_demand == 0:
                #             sum_demand = 1
                #         for k in self.fab_env.max_demand.keys():
                #             demand_feat.append(self.fab_env.total_demand[k]/sum_demand)

                state.append(demand_feat)

                self.state_lst.append(state)
                action_space = self.get_action_space()
                if self.total_setup > 0:
                    next_action = self.fab_env.dqn.get_greedy_action(self.state_lst[-1], action_space)
                else:
                    next_action = action_index
                if self.valid == False:
                    self.fab_env.dqn.train(self.state_lst, action_index, next_action, time_lst)


            else:
                total_back_order = 0
                for job in self.fab_env.job_dict.values():
                    if self in job.available_M_List:
                        total_back_order -= (self.fab_env.back_order_dic[job.id] / job.demand_time) / len(
                            job.available_M_List)

                time_lst = []
                time_lst.append(self.env.now)
                self.total_setup -= 1
                key = self.init_setup_status + "_" + action_id
                before_dem = self.fab_env.job_dict[self.init_setup_status].demand_time
                #time_lst.append(self.fab_env.job_dict[self.init_setup_status].process_time[self.name])
                self.fab_env.now_set_up[self.fab_env.job_index_dic[self.init_setup_status]] -= 1
                self.fab_env.now_set_up[self.fab_env.job_index_dic[action_id]] += 1
                self.init_setup_status = action_id
                time_lst.append(self.fab_env.set_up_time_dic[key] / before_dem)
                if self.env.now + 2 == self.fab_env.run_time or self.env.now == self.fab_env.run_time:
                    self.last_action_F(total_back_order, time_lst, action_index)
                    return 0
                yield self.env.timeout(self.fab_env.set_up_time_dic[key])
                if self.env.now + 2 == self.fab_env.run_time or self.env.now == self.fab_env.run_time:
                    self.last_action_F(total_back_order, time_lst, action_index)
                    return 0
                for i in range(self.fab_env.lot_size):
                    yield self.env.timeout(self.fab_env.job_dict[self.init_setup_status].process_time[self.name])
                    self.fab_env.inventory_dic[self.init_setup_status] += 1
                    if self.env.now + 2 == self.fab_env.run_time or self.env.now == self.fab_env.run_time:
                        self.last_action_F(total_back_order,time_lst,action_index)
                        return 0

                time_lst.append(self.env.now)

                # total_back_order = 0
                ramining_time = 0
                for job in self.fab_env.job_dict.values():
                    if self in job.available_M_List:
                        total_back_order += (self.fab_env.back_order_dic[job.id] / job.demand_time) / len(
                            job.available_M_List)
                        ramining_time += job.demand_time * self.fab_env.inventory_dic[job.id]
                        # self.fab_env.back_order_dic[job.id] = 0

                time_lst.append(total_back_order)
                time_lst.append(ramining_time)
                time_lst.append(self.fab_env.job_dict[self.init_setup_status].demand_time)
                # self.fab_env.total_cost += time_lst[1]/time_lst[0]
                self.fab_env.setup_chage_num += 1
                self.fab_env.total_cost += total_back_order
                state = []
                # state.extend(self.setup_status)
                setup_me = []
                setups = []
                for i in self.fab_env.now_set_up:
                    setups.append(i / self.fab_env.machine_num)
                # for i in range(self.fab_env.num_job):
                #     if i == self.fab_env.job_index_dic[self.init_setup_status]:
                #         setup_me.append(1)
                #     else:
                #         setup_me.append(0)

                for i in range(self.fab_env.num_job):
                    key = self.init_setup_status + "_" + self.fab_env.index_job_dic[i]
                    setup_me.append(self.fab_env.set_up_time_dic[key] / 20)
                state.append(setup_me)
                state.append(setups)
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
                demand_feat = []
                for job in self.fab_env.job_dict.values():

                    if self.name in job.process_time.keys():
                        demand_now.append(job.demand_time)
                        process_t.append(job.process_time[self.name])
                        inventory_now.append(self.fab_env.inventory_dic[job.id])
                        demand_feat.append(
                            max(0, self.fab_env.total_demand[job.id] - self.fab_env.inventory_dic[job.id]) / 100)
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
                        demand_feat.append(0)

                demand_process = []
                for job in range(len(self.fab_env.job_dict.values())):
                    demand_now[job] = demand_now[job] / 10
                    process_t[job] = process_t[job] / max_p
                    # demand_process.append(demand_now[job]/process_t[job])
                    if inventory_now[job] == -1:
                        inventory_now[job] = 1
                    else:
                        inventory_now[job] = inventory_now[job] / 100

                state.append(demand_now)
                # state.append(process_t)
                # state.extend(demand_process)
                state.append(inventory_now)

                # state.extend([self.total_setup/4])

                #         sum_demand = 0
                #         for k in self.fab_env.max_demand.keys():
                #             sum_demand += self.fab_env.total_demand[k]
                #         if sum_demand == 0:
                #             sum_demand = 1
                #         for k in self.fab_env.max_demand.keys():
                #             demand_feat.append(self.fab_env.total_demand[k]/sum_demand)

                state.append(demand_feat)
                self.state_lst.append(state)
                action_space = self.get_action_space()
                if self.total_setup > 0:
                    next_action = self.fab_env.dqn.get_greedy_action(self.state_lst[-1], action_space)
                else:
                    next_action = action_index
                if self.valid == False:
                    self.fab_env.dqn.train(self.state_lst, action_index, next_action, time_lst)

    def set_init(self, fab_env, valid,seed):
        self.fab_env = fab_env
        self.setup_status = []
        can_setup = []
        state = []
        can_job_list = []

        # for job in self.fab_env.job_dict.keys():
        #     print(self.fab_env.job_dict[job])
        # for job in self.fab_env.job_dict.values():
        #     print(job)
        # for job in self.fab_env.total_demand.keys():
        #     print(job)
        # for job in self.fab_env.max_demand.keys():
        #     print(job)

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
                random.seed(seed)
            if int(self.name[1:]) < 2:
                init_setup = random.choice(self.fab_env.group_job[0])
            elif int(self.name[1:]) < 3:
                init_setup = random.choice(self.fab_env.group_job[1])
            else:
                init_setup = random.choice(self.fab_env.group_job[2])
            print("init :; ", init_setup)
            print("machine :: ", self.name)
            self.fab_env.now_set_up[self.fab_env.job_index_dic[init_setup]] += 1
            self.setup_status[self.fab_env.job_index_dic[init_setup]] = 1
            self.init_setup_status = init_setup
            # state.extend(self.setup_status)
            setup_me = []
            setups = []
            for i in self.fab_env.now_set_up:
                setups.append(i / self.fab_env.machine_num)
            # for i in range(self.fab_env.num_job):
            #     if i == self.fab_env.job_index_dic[self.init_setup_status]:
            #         setup_me.append(1)
            #     else:
            #         setup_me.append(0)
            for i in range(self.fab_env.num_job):
                key = self.init_setup_status + "_" + self.fab_env.index_job_dic[i]
                setup_me.append(self.fab_env.set_up_time_dic[key] / 20)
            state.append(setup_me)
            state.append(setups)
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
            demand_feat = []
            for job in self.fab_env.job_dict.values():

                if self.name in job.process_time.keys():
                    demand_now.append(job.demand_time)
                    process_t.append(job.process_time[self.name])
                    inventory_now.append(self.fab_env.inventory_dic[job.id])
                    demand_feat.append(
                        max(0, self.fab_env.total_demand[job.id] - self.fab_env.inventory_dic[job.id]) / 100)
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
                    demand_feat.append(0)

            demand_process = []
            for job in range(len(self.fab_env.job_dict.values())):
                demand_now[job] = demand_now[job] / 10
                process_t[job] = process_t[job] / max_p
                # demand_process.append(demand_now[job]/process_t[job])
                if inventory_now[job] == -1:
                    inventory_now[job] = 1
                else:
                    inventory_now[job] = inventory_now[job] / 100

            state.append(demand_now)
            # state.append(process_t)
            # state.extend(demand_process)
            state.append(inventory_now)

            # state.extend([self.total_setup/4])

            #         sum_demand = 0
            #         for k in self.fab_env.max_demand.keys():
            #             sum_demand += self.fab_env.total_demand[k]
            #         if sum_demand == 0:
            #             sum_demand = 1
            #         for k in self.fab_env.max_demand.keys():
            #             demand_feat.append(self.fab_env.total_demand[k]/sum_demand)

            state.append(demand_feat)

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
        state = []
        # state.extend(self.setup_status)
        setup_me = []
        setups = []
        for i in self.fab_env.now_set_up:
            setups.append(i / self.fab_env.machine_num)
        # for i in range(self.fab_env.num_job):
        #     if i == self.fab_env.job_index_dic[self.init_setup_status]:
        #         setup_me.append(1)
        #     else:
        #         setup_me.append(0)
        for i in range(self.fab_env.num_job):
            key = self.init_setup_status + "_" + self.fab_env.index_job_dic[i]
            setup_me.append(self.fab_env.set_up_time_dic[key]/20)
        state.append(setup_me)
        state.append(setups)
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
        demand_feat = []
        for job in self.fab_env.job_dict.values():

            if self.name in job.process_time.keys():
                demand_now.append(job.demand_time)
                process_t.append(job.process_time[self.name])
                inventory_now.append(self.fab_env.inventory_dic[job.id])
                demand_feat.append(max(0, self.fab_env.total_demand[job.id] - self.fab_env.inventory_dic[job.id]) / 100)
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
                demand_feat.append(0)

        demand_process = []
        for job in range(len(self.fab_env.job_dict.values())):
            demand_now[job] = demand_now[job] / 10
            process_t[job] = process_t[job] / max_p
            # demand_process.append(demand_now[job]/process_t[job])
            if inventory_now[job] == -1:
                inventory_now[job] = 1
            else:
                inventory_now[job] = inventory_now[job] / 100

        state.append(demand_now)
        # state.append(process_t)
        # state.extend(demand_process)
        state.append(inventory_now)

        # state.extend([self.total_setup/4])

        #         sum_demand = 0
        #         for k in self.fab_env.max_demand.keys():
        #             sum_demand += self.fab_env.total_demand[k]
        #         if sum_demand == 0:
        #             sum_demand = 1
        #         for k in self.fab_env.max_demand.keys():
        #             demand_feat.append(self.fab_env.total_demand[k]/sum_demand)

        state.append(demand_feat)
        self.state_lst.append(state)
        action_space = self.get_action_space()

        if self.valid == False:
            action = self.fab_env.dqn.get_eps_action(self.state_lst[-1], action_space)
        else:
            action = self.fab_env.dqn.get_greedy_action(self.state_lst[-1], action_space)
        self.setup_status = []
        for job in self.fab_env.job_dict.values():
            self.setup_status.append(0)

        self.setup_status[action] = 1

        return action, self.fab_env.index_job_dic[action]


def scenario_generator(env, machine_num: int = 35, num_job=4, seed=0, valid=False) -> (
        dict, dict, dict, dict, dict):
    # Scenario parameters
    random.seed(2)
    min_process_time, max_process_time = (2, 3)
    min_setup_time, max_setup_time = (8, 12)
    setup_lst = [8, 10, 12, 20]
    min_demand_time, max_demand_time = (8, 10)
    min_inventory, max_inventory = (20, 50)
    back_order_dic = {}

    machine_dict = {}

    for m in range(machine_num):
        machine_name = f'M{m + 1}'
        machine_dict[machine_name] = Machine(env, machine_name, None, valid)

    inventory_dic = {}

    job_dict = {}
    job_idx = 1
    type_inx = 0
    set_up_time_dic = {}

    group_job = []

    group = []

    demand_lst = [5, 10]
    for job_type in range(num_job):
        #         available_M_num = random.randint(1, len(machine_dict.values()))
        available_M_num = machine_num
        am_list = list(machine_dict.values())

        available_M_List = random.sample(am_list, available_M_num)
        processing_time = {}

        for machine in available_M_List:
            processing_time2 = random.randint(min_process_time, max_process_time)
            processing_time2 = 2
            processing_time[machine.name] = processing_time2

        demand_time = random.choice(demand_lst)

        job_id = f'j{job_idx}'
        job = Job(env, job_id, processing_time, demand_time, available_M_List)
        job_dict[job_id] = job

        if len(group) < 4 :
            group.append(job_id)
            if len(group) == 3:
                group_job.append(copy.deepcopy(group))
                group = []


        back_order_dic[job_id] = 0
        job_idx += 1
        type_inx += 1

    #     print(group_job)

    for job1 in job_dict.values():
        for job2 in job_dict.values():
            # print(job1)
            # print(job2)
            key = job1.id + "_" + job2.id

            if job1.id == job2.id:
                set_up_time2 = 0
            else:
                if job1.id in group_job[0]:
                    if job2.id in group_job[0]:
                        set_up_time2 = 10
                    elif job2.id in group_job[1]:
                        set_up_time2 = 20
                    elif job2.id in group_job[2]:
                        set_up_time2 = 30

                elif job1.id in group_job[1]:
                    if job2.id in group_job[1]:
                        set_up_time2 = 10
                    elif job2.id in group_job[0] or job2.id in group_job[2]:
                        set_up_time2 = 20


                elif job1.id in group_job[2]:
                    if job2.id in group_job[2]:
                        set_up_time2 = 10
                    elif job2.id in group_job[1]:
                        set_up_time2 = 20
                    elif job2.id in group_job[0]:
                        set_up_time2 = 30




            #                 set_up_time2 = random.choice(setup_lst)
            #                 set_up_time2 = 10
            set_up_time_dic[key] = set_up_time2

    #     print(set_up_time_dic)

    random.seed(seed + 10)
    for job in job_dict.values():
        inventory_dic.update({job.id: random.randint(min_inventory, max_inventory)})

    return machine_dict, job_dict, set_up_time_dic, back_order_dic,group_job


class FABEnvironment:
    def __init__(self, env_type: str = 'dqn', dqn_agent=None, run_time=5000, lot_size=20):
        self.sim_env = simpy.Environment()
        self.env_type = env_type
        self.back_order_dic = {}
        self.results = []
        self.completed_job_lst = list()
        self.machine_num = 3
        self.total_cost = 0
        self.total_idle= 0
        self.lot_size = lot_size
        self.total_make = 0

        self.run_time = run_time
        self.num_job = 9

        self.min_job_num_for_a_job_type = 20
        self.max_job_num_for_a_job_type = 24

        self.dqn = dqn_agent
        self.valid_lst = []
        self.valid_setup_lst = []
        self.last_inv_lst = []
        self.seed_lst = [seed for seed in range(5000)]

    def set_scenario(self, seed, valid):
        self.now_remaing_t = []
        self.can_machine_lst = []
        self.now_set_up = []
        self.inv_list = []
        self.inventory_dic = {}
        self.total_demand = {}
        self.total_cost = 0
        self.setup_chage_num = 0
        self.total_idle = 0
        self.max_demand = {}
        self.total_make = 0
        min_demand_time, max_demand_time = (4, 80)
        demand_lst = [5, 10]
        self.machine_dict, self.job_dict, self.set_up_time_dic, self.back_order_dic,self.group_job = scenario_generator(self.sim_env,
                                                                                                         self.machine_num,
                                                                                                         self.num_job,
                                                                                                         seed=1,
                                                                                                         valid=valid)
        random.seed(seed + 100)
        min_inventory, max_inventory = (10, 30)
        for job in self.job_dict.values():
            self.inventory_dic.update({job.id: random.randint(min_inventory, max_inventory)})
            job.demand_time = random.choice(demand_lst)
            self.total_demand.update({job.id: int(960 / job.demand_time)})
            if job.demand_time == 0:
                self.max_demand.update({job.id: 0})
            else:
                self.max_demand.update({job.id: int(960 / job.demand_time)})
        self.job_index_dic = {}
        self.index_job_dic = {}
        print(self.inventory_dic," :: inv")
        # print(self.set_up_time_dic, " :: setup")
        #
        for j in self.job_dict.values():
            print(j.process_time, " :: demand")
            print(j.demand_time, " :: demand")

        index = 0

        for job in self.job_dict.values():
            self.job_index_dic.update({job.id: index})
            self.index_job_dic.update({index: job.id})
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

    def simulation_run(self, max_scenario_num=20):
        valid_make_lst = []
        if self.env_type == 'dqn':
            now = time.time()
            for scenario in range(max_scenario_num):
                seed = random.choice(self.seed_lst)
                # print("Scenario Start")
                #                 print(time.time()-now)
                now = time.time()
                if scenario >= 0:
                    valid = True
                    self.set_scenario(scenario, True)
                else:
                    valid = False
                    self.set_scenario(seed, False)
                for j in range(self.num_job):
                    self.job_dict[f'j{j + 1}'].set_fab_env(self)
                    self.sim_env.process(self.job_dict[f'j{j + 1}'].process())
                for m in range(self.machine_num):
                    self.machine_dict[f'M{m + 1}'].set_init(self, valid,(m+scenario)*3)
                    self.sim_env.process(self.machine_dict[f'M{m + 1}'].process())
                self.sim_env.run(until=self.run_time)
                self.sim_env = simpy.Environment()
                self.dqn.now_step += 1


                if  scenario >= 0:
                    inv_sum = 0
                    self.valid_lst.append(self.total_idle)
                    self.valid_setup_lst.append(self.setup_chage_num)
                    valid_make_lst.append(self.total_make)
                    #                     print(self.results, " result")
                    for invs in self.inventory_dic.values():
                        inv_sum+=invs
                    self.last_inv_lst.append(inv_sum)
                    self.inv_list.append(self.inventory_dic)
                    print(self.inv_list, " inv")
                    print(self.valid_lst, " valid")
                    print(self.valid_setup_lst, " setup")
                    print(valid_make_lst, " make")
                    print("mean inv level", np.mean(np.array(self.last_inv_lst)) / self.num_job)
                else:
                    self.results.append(self.total_idle)
            
            print(np.mean(np.array(valid_make_lst)))
            
            return self.valid_lst, self.valid_setup_lst

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


input_dim = 9
output_dim = 9
model = NN_FABSched(input_dim,output_dim)



# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4000epochs_weight0.1__3m9j_reward_demand_new_demand_960day_20.pt"))

# model.load_state_dict(torch.load("dqntest228job_0.01lr_0.999df_1000epochs_weight0.1__3m9j_noset22_new_demand_960day_20.pt"))

model.load_state_dict(torch.load("dqntest228job_0.001lr_0.99df_3000epochs_weight0.1__3m9j_noset_cpu_setup_netdemand_deuling_cnn_960day_30.pt"))

# model.load_state_dict(torch.load("dqntest228job_0.001lr_0.9df_4000epochs_weight0.1__3m9j_reward_inv_new_demand_960day_20.pt"))

dqn = DQN_FS(model,learning_rate=10**(-3),env_name="test")
lot_size = 30
#
#
env = FABEnvironment(env_type='dqn',dqn_agent=dqn,run_time=960,lot_size=lot_size)
# # env.set_scenario()
env.simulation_run()

