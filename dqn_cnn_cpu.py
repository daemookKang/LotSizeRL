from datetime import datetime
import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter
# from state import MultipleMachineState
# from torch_geometric.data import Data
import numpy as np
import copy
# from graph_lib import *

import datetime


class DQN_FS:
    def __init__(self, model=None, learning_rate=10 ** (-5), decay_facotor=0.99, Min_suard_gradient=10 ** (-6),
                 epsilon=0.6, e_decay=10 ** (-4)
                 , replay_buffer_size=10 ** 5, batch_size=64, discount_rate=0.9, train_step=20, weight_inv=0.1,
                 env_name="3"):
        #         self.device = torch.device('cuda')

        self.model = model
        print("aa")
        self.target = copy.deepcopy(self.model)
        self.env_name = env_name
        self.learning_rate = learning_rate
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)
        # self.model.load_state_dict(torch.load("dqntest2_1000epochs.pt"))
        self.eps = epsilon
        self.weight_inv = weight_inv
        decayRate = 0.96
        self.last_eps = 0.01
        self.my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

        self.e_decay = e_decay
        self.batch_size = batch_size
        self.discount_factor = discount_rate
        self.now_step = 0
        self.update_step = train_step
        # self.loss_lst = []
        self.replay_buffer = deque(maxlen=int(replay_buffer_size/4)*5)
        self.fir = True

    def decaying_eps(self):
        r = max(0, (3000 - self.now_step) / 3000)
        self.eps = (self.eps - self.last_eps) * r + self.last_eps

    def get_greedy_action(self, state, action_space):
        state = self.list_2_torch(state)
        q_value = self.model(state)
        action_space = action_space  # env랑 맞추기
        max_val = -9999999999
        chk = 0
        # print(q_value)

        for i in range(len(action_space)):
            # print(q_value[i] * action_space[i])
            if q_value[i] * action_space[i] > max_val and action_space[i] != 0:
                max_val = q_value[i]
                greedy_action = i
        #     q_value_list = dict()
        #     for action in action_space:
        #
        #         q_value_list[action] = self.model(input)
        #
        #     greedy_action = min(q_value_list,key=q_value_list.get)
        # print(chk)

        return greedy_action

    def list_2_torch(self, input_list):
        return torch.FloatTensor(input_list)

    # numpy 형식이면 더 쉽게도 가능하나 일단 코딩
    def get_random_action(self, action_space):
        action_space = action_space
        real_action = []

        for i in range(len(action_space)):
            if action_space[i] == 1:
                real_action.append(i)

        rand_action = random.choice(real_action)
        return rand_action

    def get_eps_action(self, state, action_space):
        p = np.random.rand()
        if p < self.eps:
            return self.get_random_action(action_space)
        else:
            return self.get_greedy_action(state, action_space)

    def get_q_value(self, state, action_space):
        state = self.list_2_torch(state)
        q_value = self.model(state)
        action_space = action_space  # env랑 맞추기
        max_val = -9999999999
        chk = 0
        print(q_value)

        for i in range(len(action_space)):
            # print(q_value[i] * action_space[i])
            if q_value[i] * action_space[i] > max_val and action_space[i] != 0:
                max_val = q_value[i]
                greedy_action = i

        return max_val

    def get_sars(self, state_lst, action, action_space, time_lst):
        sars_lst = []
        # state_lst = []
        # action_lst = []
        # reward_lst = []

        this_state = state_lst[-2]
        next_state = state_lst[-1]
        this_action = action
        action_space = action_space
        discount_step = time_lst[2] - time_lst[0]
        deamand_feats = []
        # print(this_state)
        for i in range(len(this_state[-1])):
            deamand_feats.append(this_state[-1][i])
        # print(deamand_feats)
        add_demand = 0
        if deamand_feats[this_action] <= 0:
            add_demand = -5
        #             print(deamand_feats)
        #             print(this_action)
        #             print(deamand_feats[this_action])
        else:
            add_demand = 0

        # if this_action == 'idle':
        #     process_time = 0
        # else:
        #     process_time = env.job_dict[this_action].cur_process_time
        reward = self.get_reward(time_lst, add_demand)

        sars = this_state, this_action, reward, next_state, action_space, discount_step

        sars_lst.append(sars)

        # for idx, state in enumerate(env.state_lst[:-1]):
        #     this_state = env.state_lst[idx]
        #     this_action = env.action_lst[idx]
        #     if this_action == 'idle':
        #         process_time = 0
        #     else:
        #         process_time = env.job_dict[this_action].process_time
        #
        #     next_state = env.state_lst[idx+1]
        #
        #     reward = self.get_reward(this_state, next_state,process_time)
        #     state_lst.append(this_state)
        #     action_lst.append(this_action)
        #     reward_lst.append(reward)
        #     next_action = env.action_lst[idx+1]
        #
        #
        #     sars_lst.append(sars)

        return sars_lst

    def get_reward(self, time_lst, add_demand):
        # reward = -1*(time_lst[4]+time_lst[5]*self.weight_inv)
        # reward = (-1 * (time_lst[3]) + add_demand) / 30
        reward = (-1 * (time_lst[1] / 10 + time_lst[3]) + add_demand) / 30
        #         reward = (-1 * ( time_lst[3]) + add_demand)/30
        #         print(time_lst[3], " :: Re")
        return reward

    def train(self, state_lst, action, next_action, time_lst):

        sars_lst = self.get_sars(state_lst, action, next_action, time_lst)

        self.replay_buffer += sars_lst

        # Train network

        if len(self.replay_buffer) >= self.batch_size:
            self.net_train()
        # Copy trained network to target network
        # if self.now_step % self.update_step == 0:
        #     self.target.load_state_dict(self.model.state_dict())

    def update_dqn(self):
        self.target.load_state_dict(self.model.state_dict())
        #         self.eps = max(0.01,(self.eps - 0.0005)) #linear?? or multiple??

        self.my_lr_scheduler.step()

    def save_model(self):
        torch.save(self.model.state_dict(),
                   'dqntest_lowmodel_job_{}lr_{}df_{}epochs_weight{}_{}.pt'.format(self.learning_rate,
                                                                                   self.discount_factor, self.now_step,
                                                                                   self.weight_inv, self.env_name))

    def net_train(self):
        self.optimizer.zero_grad()
        batch = random.sample(self.replay_buffer, self.batch_size)  # Get batch data from replay buffer
        loss_t = self.cal_loss(batch)
        if self.now_step % 5 == 0:
            print(f'Step: {self.now_step}, Loss: {loss_t.item()}')
        #         print(loss_t)

        if self.now_step % 1000 == 0:
            # print(f'Step: {self.now_step}, Loss: {loss_t.item()}')
            torch.save(self.model.state_dict(),
                       'dqntest228job_{}lr_{}df_{}epochs_weight{}_{}.pt'.format(self.learning_rate,
                                                                                self.discount_factor, self.now_step,
                                                                                self.weight_inv, self.env_name))

        if self.now_step % 500 == 0:
            if self.fir == True:
                # self.loss_lst.append(loss_t)
                # torch.save(self.model.state_dict(), 'dqntest_{}lr_{}df_{}epochs.pt'.format(self.learning_rate,self.discount_factor, self.now_step))
                self.fir = False
        else:
            self.fir = True

        loss_t.backward()
        self.optimizer.step()

    def cal_loss(self, batch):
        x = []
        y = []

        for data in batch:
            state, action, reward, next_state, action_space, discount_step = data
            #             print(state, " :: state")
            #             print(next_state, " :: next_state")
            #             print(action, " :: action")
            #             print(reward, " :: reward")
            #             print(action_space, " :: as")

            # print(data)

            sa_input = self.list_2_torch(state)
            nsa_input = self.list_2_torch(next_state)

            # print(self.model(sa_input),"model")
            q = self.model(sa_input)[action]

            q_t = self.model(nsa_input)
            max_val = -9999999999
            next_greedy_action = -99999999

            if action_space == 'no':
                done = True
                # print(state)
                # print(action)
                # print(reward)
                # print("AAA")
            else:
                done = False
                for i in range(len(action_space)):

                    if q_t[i] * action_space[i] > max_val and action_space[i] != 0:
                        max_val = q_t[i]
                        next_greedy_action = i

            # print(next_state,"!@!@!@")

            #             print(q, " :: q_val")
            # print(self.model(sa_input).gather(1,action))
            # print(state)
            # print(next_state)
            # print(next_greedy_action)
            # print(reward)
            # print(action)

            if len(nsa_input) != 0:
                if done == True:
                    tgt_q = torch.as_tensor(reward)
                else:
                    tgt_q = reward + (
                                (self.discount_factor ** discount_step) * self.target(nsa_input)[next_greedy_action])

                #                 print((self.discount_factor**discount_step))
                # print(next_greedy_action)
                # print(self.target(nsa_input)[next_greedy_action],"@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
                x.append(q)
                # print(tgt_q)
                y.append(tgt_q)

        x = torch.stack(x)
        # print(len(y))
        # x.requires_grad = True
        # print(x, "x")
        # print(y, "y")
        y = torch.stack(y)

        return nn.MSELoss()(x, y)

    # def last_state(self,env):
    #
    #     for i in env.machine_list:
    #         sars_lst = []
    #         this_state = env.state_lst[i][0]
    #         next_state = env.state_lst[i][1]
    #         this_action = env.action_lst[i]
    #         next_greedy_action = 'no'
    #         sars = this_state, this_action, reward, next_state, next_greedy_action
    #
    #         sars_lst.append(sars)
    #         self.replay_buffer += sars_lst
