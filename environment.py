import simpy
import numpy as np
import random
import copy
from dqn import DQN_FS
from nn import NN_FABSched
import matplotlib.pyplot as plt



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
        while True:
            if self.env.now-1 == self.fab_env.run_time:
                return 0
            self.fab_env.inventory_dic[self.id] -= 1


            if self.fab_env.inventory_dic[self.id] < 0:
                self.fab_env.inventory_dic[self.id] = 0
                self.fab_env.back_order_dic[self.id] += 1
                yield self.env.timeout(1)
            else:
                self.fab_env.total_demand[self.id] -= 1
                # self.fab_env.back_order_dic[self.id] = 0
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
        self.total_setup = 100000000

        self.last_action = True  # 마지막 action이 idle이었으면 False



    def __str__(self):
        return f'name: {self.name}\nsetup_status: {self.setup_status}\n'

    def process(self):
        while True:
            if self.env.now-1 == self.fab_env.run_time:
                return 0



            if self.total_setup > 0:
                action_index,action_id = self.get_action()
            # print(action_id , " :: action")
            if action_index == self.fab_env.job_index_dic[self.init_setup_status]:

                time_lst = []
                time_lst.append(self.env.now)
                total_back_order = 0
                for job in self.fab_env.job_dict.values():
                    if self in job.available_M_List:
                        total_back_order-=(self.fab_env.back_order_dic[job.id]/ job.demand_time)/len(job.available_M_List)
                for i in range(self.fab_env.lot_size):
                    yield self.env.timeout(self.fab_env.job_dict[self.init_setup_status].process_time[self.name])
                    self.fab_env.inventory_dic[self.init_setup_status] += 1
                # print(self.fab_env.inventory_dic, " :: action")

                time_lst.append(1)
                time_lst.append(self.env.now)
                time_lst.append(0)
                # total_back_order = 0
                ramining_time = 0
                for job in self.fab_env.job_dict.values():
                    if self in job.available_M_List:
                        total_back_order += (self.fab_env.back_order_dic[job.id]/ job.demand_time)/len(job.available_M_List)
                        ramining_time += job.demand_time*self.fab_env.inventory_dic[job.id]
                        # self.fab_env.back_order_dic[job.id] = 0

                time_lst.append(total_back_order)
                time_lst.append(ramining_time)
                time_lst.append(self.fab_env.job_dict[self.init_setup_status].demand_time)
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
                        inventory_now.append(self.fab_env.inventory_dic[job.id])
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
                demand_process = []
                for job in range(len(self.fab_env.job_dict.values())):
                    demand_now[job] = demand_now[job] / 10
                    process_t[job] = process_t[job] / max_p
                    # demand_process.append(demand_now[job]/process_t[job])
                    if inventory_now[job] == -1:
                        inventory_now[job] =  1
                    else:
                        inventory_now[job] = inventory_now[job] / 100

                state.extend(demand_now)
                state.extend(process_t)
                # state.extend(demand_process)
                state.extend(inventory_now)
                # state.extend([self.total_setup/4])
                demand_feat = []
                sum_demand = 0
                for k in self.fab_env.max_demand.keys():
                    sum_demand += self.fab_env.total_demand[k]
                if sum_demand == 0:
                    sum_demand = 1
                for k in self.fab_env.max_demand.keys():
                    demand_feat.append(self.fab_env.total_demand[k] / self.fab_env.max_demand[k])

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
                total_back_order = 0
                for job in self.fab_env.job_dict.values():
                    if self in job.available_M_List:
                        total_back_order -= (self.fab_env.back_order_dic[job.id] / job.demand_time) / len(job.available_M_List)
                time_lst = []
                time_lst.append(self.env.now)
                self.total_setup -= 1
                key = self.init_setup_status + "_" + action_id
                before_dem = self.fab_env.job_dict[self.init_setup_status].demand_time
                time_lst.append(self.fab_env.job_dict[self.init_setup_status].process_time[self.name])
                self.fab_env.now_set_up[self.fab_env.job_index_dic[self.init_setup_status]] -= 1
                self.fab_env.now_set_up[self.fab_env.job_index_dic[action_id]] += 1
                self.init_setup_status = action_id
                yield self.env.timeout(self.fab_env.set_up_time_dic[key])

                for i in range(self.fab_env.lot_size):


                    yield self.env.timeout(self.fab_env.job_dict[self.init_setup_status].process_time[self.name])
                    self.fab_env.inventory_dic[self.init_setup_status] += 1

                time_lst.append(self.env.now)
                time_lst.append(self.fab_env.set_up_time_dic[key]/before_dem)
                # total_back_order = 0
                ramining_time = 0
                for job in self.fab_env.job_dict.values():
                    if self in job.available_M_List:
                        total_back_order+=(self.fab_env.back_order_dic[job.id] / job.demand_time) / len(job.available_M_List)
                        ramining_time += job.demand_time * self.fab_env.inventory_dic[job.id]
                        # self.fab_env.back_order_dic[job.id] = 0

                time_lst.append(total_back_order)
                time_lst.append(ramining_time)
                time_lst.append(self.fab_env.job_dict[self.init_setup_status].demand_time)
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
                demand_process = []
                for job in range(len(self.fab_env.job_dict.values())):
                    demand_now[job] = demand_now[job] / 10
                    process_t[job] = process_t[job] / max_p
                    # demand_process.append(demand_now[job]/process_t[job])
                    if inventory_now[job] == -1:
                        inventory_now[job] =  1
                    else:
                        inventory_now[job] = inventory_now[job] / 100
                state.extend(demand_now)
                state.extend(process_t)

                # state.extend(demand_process)
                state.extend(inventory_now)


                # state.extend([self.total_setup/4])
                demand_feat = []
                sum_demand = 0
                for k in self.fab_env.max_demand.keys():
                    sum_demand += self.fab_env.total_demand[k]
                if sum_demand == 0:
                    sum_demand = 1
                for k in self.fab_env.max_demand.keys():
                    demand_feat.append(self.fab_env.total_demand[k] / self.fab_env.max_demand[k])

                state.extend(demand_feat)
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
                random.seed(1)
            init_setup = random.choice(can_job_list)
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
            demand_process = []
            for job in range(len(self.fab_env.job_dict.values())):
                demand_now[job] = demand_now[job] / 10
                process_t[job] = process_t[job] / max_p
                # demand_process.append(demand_now[job]/process_t[job])
                if inventory_now[job] == -1:
                    inventory_now[job] =  1
                else:
                    inventory_now[job] = inventory_now[job] / 100

            state.extend(demand_now)
            state.extend(process_t)
            # state.extend(demand_process)
            state.extend(inventory_now)

            # state.extend([self.total_setup/4])
            demand_feat = []
            sum_demand = 0
            for k in self.fab_env.max_demand.keys():
                sum_demand += self.fab_env.total_demand[k]
            if sum_demand == 0:
                sum_demand = 1
            for k in self.fab_env.max_demand.keys():
                demand_feat.append(self.fab_env.total_demand[k] / self.fab_env.max_demand[k])

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

        demand_process=[]
        for job in range(len(self.fab_env.job_dict.values())):
            demand_now[job] = demand_now[job] / 10
            process_t[job] = process_t[job] / max_p
            # demand_process.append(demand_now[job]/process_t[job])
            if inventory_now[job] == -1:
                inventory_now[job] = 1
            else:
                inventory_now[job] = inventory_now[job] / 100

        state.extend(demand_now)
        state.extend(process_t)
        # state.extend(demand_process)
        state.extend(inventory_now)

        # state.extend([self.total_setup/4])

        demand_feat = []
        sum_demand = 0
        for k in self.fab_env.max_demand.keys():
            sum_demand += self.fab_env.total_demand[k]
        if sum_demand == 0:
            sum_demand = 1
        for k in self.fab_env.max_demand.keys():
            demand_feat.append(self.fab_env.total_demand[k]/self.fab_env.max_demand[k])

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
    def __init__(self, env_type: str = 'dqn', dqn_agent=None,run_time=5000, lot_size=20):
        self.sim_env = simpy.Environment()
        self.env_type = env_type
        self.back_order_dic = {}
        self.results = []
        self.completed_job_lst = list()
        self.machine_num = 14
        self.total_cost = 0
        self.lot_size = lot_size

        self.run_time=run_time
        self.num_job = 45

        self.min_job_num_for_a_job_type = 20
        self.max_job_num_for_a_job_type = 24

        self.dqn = dqn_agent
        self.valid_lst = []
        self.valid_setup_lst = []
        self.seed_lst = [seed for seed in range(5000)]

    def set_scenario(self,seed,valid):
        self.now_remaing_t = []
        self.can_machine_lst = []
        self.now_set_up = []
        self.inventory_dic = {}
        self.total_demand = {}
        self.total_cost = 0
        self.setup_chage_num = 0
        self.max_demand = {}
        min_demand_time, max_demand_time = (4, 10)
        demand_lst = [5,6,7,8,9,10]
        self.machine_dict,  self.job_dict , self.set_up_time_dic,self.back_order_dic = scenario_generator(self.sim_env, self.machine_num, self.num_job,seed = 1,valid=valid)
        random.seed(seed + 100)
        min_inventory, max_inventory = (10, 80)
        for job in self.job_dict.values():
            self.inventory_dic.update({job.id: random.randint(min_inventory, max_inventory)})
            job.demand_time = random.choice(demand_lst)
            self.total_demand.update({job.id:int(960/job.demand_time)})
            if job.demand_time == 0:
                self.max_demand.update({job.id: 0})
            else:
                self.max_demand.update({job.id:int(960/job.demand_time)})
        self.job_index_dic = {}
        self.index_job_dic = {}
        # print(self.inventory_dic," :: inv")
        # print(self.set_up_time_dic, " :: setup")
        #
        # for j in self.job_dict.values():
        #     print(j.process_time, " :: demand")
        #     print(j.demand_time, " :: demand")

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

    def simulation_run(self,  max_scenario_num=4002):
        if self.env_type == 'dqn':
            for scenario in range(max_scenario_num):
                seed =random.choice(self.seed_lst)
                if scenario % 51 == 0 and scenario > 0:
                    valid = True
                    self.set_scenario(50000,True)
                else:
                    valid = False
                    self.set_scenario(seed, False)
                for j in range(self.num_job):
                    self.job_dict[f'j{j+1}'].set_fab_env(self)
                    self.sim_env.process(self.job_dict[f'j{j+1}'].process())
                for m in range(self.machine_num):
                    self.machine_dict[f'M{m + 1}'].set_init(self,valid)
                    self.sim_env.process(self.machine_dict[f'M{m + 1}'].process())
                self.sim_env.run(until=self.run_time)
                self.sim_env = simpy.Environment()
                self.dqn.now_step +=1
                if scenario%10 == 0 and scenario>0:
                    self.dqn.update_dqn()

                if scenario % 51 ==0 and scenario >0:
                    self.valid_lst.append(self.total_cost)
                    self.valid_setup_lst.append(self.setup_chage_num)
                    print(self.results, " result")
                    print(self.valid_lst , " valid")
                    print(self.valid_setup_lst, " setup")
                else:
                    self.results.append(self.total_cost)

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

#
input_dim = 225
output_dim = 45
lot_size_lst = [20,30]
lr_lst = [10**(-2),10**(-3)]
df_lst = [0.99,0.9,0.8,0.7]
valid_lst = {}
setup_lst ={}
for lot_size in lot_size_lst:
    for lr in lr_lst:
        for df in df_lst:
            model = NN_FABSched(input_dim,output_dim)
            #
            dqn = DQN_FS(model,learning_rate=lr,discount_rate=df,env_name="_14m45j_noset22_new_backorder_960day_{}".format(lot_size))
            #
            #
            env = FABEnvironment(dqn_agent=dqn,run_time=960,lot_size=lot_size)
            # # env.set_scenario()
            valid, setup = env.simulation_run()
            valid_lst.update({str(lr)+"_"+str(df)+"_"+str(lot_size): valid})
            setup_lst.update({str(lr)+"_"+str(df)+"_"+str(lot_size): setup})



print(valid_lst)
print(setup_lst)

