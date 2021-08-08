import simpy
import numpy as np
import random
import copy
from dqn import DQN_FS
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
        self.next_demand = 4
        self.chk = 1

    def __str__(self):
        return f'id: {self.id}\navailable_M_List: {self.available_M_List}\nprocess_time: {self.process_time}\ndemand_time: {self.demand_time}\n'

    def change_demand(self):

        print(self.id)
        random.seed(self.chk + int(self.id[1]))
        print("now deamnd :; ", self.demand_time)
        demand_lst = [5,10]
        self.demand_time = copy.deepcopy(self.next_demand)
        self.next_demand = random.choice(demand_lst)
        print("after deamnd :; ", self.demand_time)
        print(self.env.now, " :: now")
        self.change = True
        self.chk += 1

    def get_processing_time(self,mac):
        return self.process_time[mac]

    def set_fab_env(self,fab_env):
        self.fab_env = fab_env

    def process(self):
        chk =1
        while True:


            if self.env.now-1 == self.fab_env.run_time:
                return 0
            self.fab_env.inventory_dic[self.id] -= 1


            if self.fab_env.inventory_dic[self.id] < 0:
                self.fab_env.inventory_dic[self.id] = 0
                self.fab_env.back_order_dic[self.id] += 1
                self.fab_env.total_idle += 1
                yield self.env.timeout(1)
            else:
                self.fab_env.total_demand[self.id] -= 1
                # self.fab_env.back_order_dic[self.id] = 0
                self.fab_env.total_make += 1
                yield self.env.timeout(self.demand_time)
                #count back-order
            # print(self.fab_env.inventory_dic , " :: inventory")
            # print(self.env.now, " :: now")



class Machine:
    def __init__(self, env=None, name: str = 'M1', init_setup_status=None,valid=False):
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
        self.total_setup = 10000000000000000000

        self.last_action = True  # 마지막 action이 idle이었으면 False



    def __str__(self):
        return f'name: {self.name}\nsetup_status: {self.setup_status}\n'

    def process(self):
        chk = 0
        while True:
            if self.env.now-1 == self.fab_env.run_time:
                return 0

            # if chk * 1440 <= self.env.now:
            #     for job in self.fab_env.job_dict.keys():
            #
            #         if self.env.now > 0:
            #             self.fab_env.job_dict[job].change_demand()
            #         for job in self.fab_env.job_dict.values():
            #             # self.fab_env.total_demand.update({job.id: int(1440 / job.demand_time) + int(1440 / job.next_demand)})
            #             # self.fab_env.max_demand.update({job.id: int(1440 / job.demand_time) + int(1440 / job.next_demand)})
            #             self.fab_env.total_demand.update({job.id: int(960 / job.demand_time) })
            #             self.fab_env.max_demand.update({job.id: int(960 / job.demand_time) })
            #
            #     chk+=1

            if self.total_setup > 0:
                action_index,action_id = self.get_action()
            # print(action_id , " :: action")
            print(self.init_setup_status , " -- > " ,action_id,"  :: ",self.name)
            print(self.fab_env.inventory_dic)
            if action_index == self.fab_env.job_index_dic[self.init_setup_status]:

                time_lst = []

                for i in range(self.fab_env.lot_size):
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
                        inventory_now.append(self.fab_env.inventory_dic[job.id] )
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
                        inventory_now[job] = inventory_now[job] / 100

                state.extend(demand_now)

                state.extend(process_t)
                state.extend(inventory_now)
                # state.extend([self.total_setup/4])
                demand_feat = []
                sum_demand = 0
                for k in self.fab_env.max_demand.keys():
                    sum_demand += self.fab_env.total_demand[k]
                if sum_demand == 0:
                    sum_demand = 1
                for k in self.fab_env.max_demand.keys():
                    demand_feat.append(self.fab_env.total_demand[k] / sum_demand)

                state.extend(demand_feat)

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

                for i in range(self.fab_env.lot_size):


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
                        inventory_now.append(self.fab_env.inventory_dic[job.id] )
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
                        inventory_now[job] = inventory_now[job] / 100

                state.extend(demand_now)

                state.extend(process_t)
                state.extend(inventory_now)
                # state.extend([self.total_setup/4])
                demand_feat = []
                sum_demand = 0
                for k in self.fab_env.max_demand.keys():
                    sum_demand += self.fab_env.total_demand[k]
                if sum_demand == 0:
                    sum_demand = 1
                for k in self.fab_env.max_demand.keys():
                    demand_feat.append(self.fab_env.total_demand[k] / sum_demand)

                state.extend(demand_feat)
                self.state_lst.append(state)
                action_space = self.get_action_space()
                if self.total_setup > 0:
                    next_action = self.fab_env.dqn.get_greedy_action(self.state_lst[-1], action_space)
                else:
                    next_action = action_index
                if self.valid == False:
                    self.fab_env.dqn.train(self.state_lst,action_index,next_action,time_lst)



    def set_init(self,fab_env,valid,seed):
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
                random.seed(seed)
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
                    inventory_now.append(self.fab_env.inventory_dic[job.id] )
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
                    inventory_now[job] = inventory_now[job] / 100

            state.extend(demand_now)

            state.extend(process_t)
            state.extend(inventory_now)
            # state.extend([self.total_setup/4])
            demand_feat = []
            sum_demand = 0
            for k in self.fab_env.max_demand.keys():
                sum_demand += self.fab_env.total_demand[k]
            if sum_demand == 0:
                sum_demand = 1
            for k in self.fab_env.max_demand.keys():
                demand_feat.append(self.fab_env.total_demand[k] / sum_demand)

            state.extend(demand_feat)

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
                    inventory_now.append(self.fab_env.inventory_dic[job.id] )
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
                    inventory_now[job] = inventory_now[job] / 100

            state.extend(demand_now)

            state.extend(process_t)
            state.extend(inventory_now)
            # state.extend([self.total_setup/4])
            demand_feat = []
            sum_demand = 0
            for k in self.fab_env.max_demand.keys():
                sum_demand += self.fab_env.total_demand[k]
            if sum_demand == 0:
                sum_demand = 1
            for k in self.fab_env.max_demand.keys():
                demand_feat.append(self.fab_env.total_demand[k] /sum_demand)

            state.extend(demand_feat)

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




def scenario_generator(env, machine_num: int = 35,num_job=4,seed = 0,valid=False) -> (
dict, dict, dict, dict, dict):
    # Scenario parameters
    random.seed(2)
    min_process_time, max_process_time = (2, 3)
    min_setup_time, max_setup_time = (8, 10)
    min_demand_time, max_demand_time = (8, 10)
    min_inventory,max_inventory = (20,50)
    back_order_dic = {}




    machine_dict = {}



    for m in range(machine_num):
        machine_name = f'M{m + 1}'
        machine_dict[machine_name] = Machine(env, machine_name,None,valid)


    inventory_dic = {}

    job_dict = {}
    job_idx = 1
    type_inx = 0
    set_up_time_dic = {}
    demand_lst = [5, 10]
    for job_type in range(num_job):
        available_M_num = random.randint(1, len(machine_dict.values()))
        am_list = list(machine_dict.values())

        available_M_List = random.sample(am_list, available_M_num)
        processing_time = {}


        for machine in available_M_List:
            processing_time2 = random.randint(min_process_time, max_process_time)
            processing_time2 = 2
            processing_time[machine.name] = processing_time2

        demand_time = random.choice(demand_lst)



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
    def __init__(self, env_type: str = 'dqn', dqn_agent=None,run_time=5000,lot_size=20):
        self.sim_env = simpy.Environment()
        self.env_type = env_type
        self.back_order_dic = {}
        self.results = []
        self.completed_job_lst = list()
        self.machine_num = 3
        self.total_cost = 0
        self.total_make = 0
        self.total_idle = 0
        self.lot_size = lot_size
        self.inv_lst = []
        self.run_time=run_time
        self.num_job = 9

        self.min_job_num_for_a_job_type = 20
        self.max_job_num_for_a_job_type = 24

        self.dqn = dqn_agent
        self.valid_lst = []
        self.valid_setup_lst = []
        self.valid_make_lst = []
        self.seed_lst = [seed for seed in range(500)]

    def set_scenario(self,seed,valid,init_inv=None):

        self.now_remaing_t = []
        self.can_machine_lst = []
        self.now_set_up = []
        self.inventory_dic = {}
        self.total_demand = {}
        self.max_demand = {}
        self.total_cost = 0
        self.setup_chage_num = 0
        self.total_make = 0
        self.total_idle = 0
        min_demand_time, max_demand_time = (4,5)
        demand_lst = [5,10]
        self.machine_dict,  self.job_dict , self.set_up_time_dic,self.back_order_dic = scenario_generator(self.sim_env, self.machine_num, self.num_job,seed = 1,valid=valid)
        random.seed(seed + 100)
        min_inventory, max_inventory = (10, 80)
        for job in self.job_dict.values():
            self.inventory_dic.update({job.id: random.randint(min_inventory, max_inventory)})
            job.demand_time = random.choice(demand_lst)
            # job.next_demand = random.choice(demand_lst)
            self.total_demand.update({job.id:int(960/job.demand_time)})
            self.max_demand.update({job.id: int(960 / job.demand_time)})
        if init_inv != None and init_inv != 'test':
            self.inventory_dic = init_inv

        lstInitInv = [20, 26, 37, 25]
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
            inv_list = []
            demand_list = []
            for scenario in range(max_scenario_num):
                seed =random.choice(self.seed_lst)
                if  scenario >= 0:

                    valid = True
                    self.set_scenario(scenario,True)
                    inv = []
                    dem = []
                    for i in self.inventory_dic.keys():
                        inv.append(self.inventory_dic[i])
                        dem.append(self.job_dict[i].demand_time)
                    inv_list.append(inv)
                    demand_list.append(dem)
                else:
                    valid = True
                    self.set_scenario(scenario,True,self.inventory_dic)

                for j in range(self.num_job):
                    self.job_dict[f'j{j+1}'].set_fab_env(self)
                    self.sim_env.process(self.job_dict[f'j{j+1}'].process())
                for m in range(self.machine_num):
                    self.machine_dict[f'M{m + 1}'].set_init(self,valid,1)
                    self.sim_env.process(self.machine_dict[f'M{m + 1}'].process())
                self.sim_env.run(until=self.run_time)
                self.sim_env = simpy.Environment()
                self.dqn.now_step +=1
                # if scenario%10 == 0 and scenario>0:
                #     self.dqn.update_dqn()

                if scenario >= 0:

                    self.valid_lst.append(self.total_idle)
                    self.valid_setup_lst.append(self.setup_chage_num)
                    self.valid_make_lst.append(self.total_make)
                    sums = 0
                    for invs in self.inventory_dic.values():
                        sums += invs
                    self.inv_lst.append(sums)
                    print(sums, " inv")
                    print(self.results, " result")
                    print(self.valid_lst , " valid")
                    print(self.valid_setup_lst, " setup")
                    print("mean = ",np.mean(np.array(self.valid_lst)))
                    print(self.total_make, " :: make")
                    print(self.total_idle, " :: idle")
                else:
                    self.results.append(self.total_cost)

            print("idle_lst ", self.valid_lst)
            print("total_make ", self.valid_make_lst)
            print("setup ", self.valid_setup_lst)
            print("inv_lst = ", inv_list)
            print("dem_lst = ", demand_list)
            print("mean_make = ", np.mean(np.array(self.valid_make_lst)))
            print("mean inv = ", np.mean(np.array(self.inv_lst)))

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
                    self.valid_lst.append(self.total_idle)
                    self.valid_setup_lst.append(self.setup_chage_num)
                    self.valid_make_lst.append(self.total_make)

                    print(self.valid_lst , " valid")
                    print(self.valid_setup_lst, " setup")
                    print(self.total_make, " :: make")
                    print(self.total_idle, " :: idle")
                    print("mean_idle = ",np.mean(np.array(self.valid_lst)))
                    print("mean_make = ", np.mean(np.array(self.valid_make_lst)))

                else:
                    self.results.append(self.total_cost)



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
import time

input_dim = 45
output_dim = 9
model = NN_FABSched(input_dim,output_dim)



# model.load_state_dict(torch.load("dqntest228job_0.01lr_0.7df_4000epochs_weight0.1__3m9j_noset22_new_backorder_960day_20.pt"))
#
model.load_state_dict(torch.load("dqntest228job_0.01lr_0.99df_1000epochs_weight0.1__3m9j_noset22_new_demand_960day_20.pt"))

# model.load_state_dict(torch.load("dqntest228job_0.01lr_0.7df_4000epochs_weight0.1__3m9j_noset22_new_demand_960day_20.pt"))

dqn = DQN_FS(model,learning_rate=10**(-3),env_name="test")
lot_size = 20
#
#
env = FABEnvironment(env_type='dqn',dqn_agent=dqn,run_time=960,lot_size=lot_size)
# # env.set_scenario()
env.simulation_run()



