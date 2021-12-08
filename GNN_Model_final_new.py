import numpy as np
import pandas as pd
import torch
import random
import torch_geometric as tg
import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, remove_self_loops, softmax
from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,OptTensor)
from torch import Tensor
import torch.nn as nn
from torch.nn import Parameter, Linear, ReLU
from torch_geometric.nn.inits import glorot, zeros
from torch_sparse import SparseTensor, set_diag
import torch.nn.functional as F
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class EmbeddingPrecedence(MessagePassing):
    def __init__(self, num_feats):
        super(EmbeddingPrecedence, self).__init__(aggr='add')

    def forward(self, g):
        return self.propagate(g.edge_index_P, x=g.x, c=g.c)

    def message(self, c_j):

        return c_j



class EmbeddingMachine2(MessagePassing):
    def __init__(self, num_feats):
        super(EmbeddingMachine2, self).__init__(aggr='add')

    def forward(self, g,m):

        PT = g.index_PT[m-1]
        m_edge = g.index_m[m-1].t()

        reverse_edge = [m_edge[1].tolist(),m_edge[0].tolist()]

        reverse_edge = torch.tensor(reverse_edge , dtype=torch.long)


        return self.propagate(reverse_edge, x=g.x, notj = g.notj, pt=PT)

    def message(self,notj_j, pt):

        # print(pt_j)
        return pt*notj_j

class EmbeddingMachinemu2(MessagePassing):
    def __init__(self, num_feats):
        super(EmbeddingMachinemu2, self).__init__(aggr='add')


    def forward(self, g,m):
        m_edge = g.index_m[m-1].t()
        mu2 = g.mu2


        return self.propagate(m_edge, x=g.x, mu2=mu2)

    def message(self, mu2_j):

        # print(pt_j)
        return mu2_j

class EmbedC(MessagePassing):
    def __init__(self, num_feats):
        super(EmbedC, self).__init__(aggr='add')


    def forward(self, g , e):
        c = g.c

        # print(c)
        # print(e)

        return self.propagate(e, x=g.x, c=c)

    def message(self,c_j):

        return c_j




class Net_MLP(nn.Module):
    def __init__(self, num_feats):
        super(Net_MLP, self).__init__()
        self.num_feats = num_feats
        self.embeddingPrecedence = EmbeddingPrecedence(num_feats)
        self.embeddingMachinemu2 = EmbeddingMachinemu2(num_feats)
        self.embeddingMachine2 = EmbeddingMachine2(num_feats)
        self.embedC = EmbedC(num_feats)
        self.theta_v = nn.Linear(1, num_feats, bias=False).to(device)
        self.theta_v2 = nn.Linear(num_feats, num_feats, bias=False).to(device)
        self.theta_v3 = nn.Linear(num_feats, 1, bias=False).to(device)


        self.relu = nn.ReLU().to(device)


    def forward(self, g,v, e,m,batch_size=1, Train = True):
        c1  = self.embedC(g,e[0])
        g.notj[v] = 0
        if len(g.edge_index_P) == 0:
            c2 = []
        else:
            c2 = self.embeddingPrecedence(g)



        g.mu2 = self.theta_v(self.embeddingMachine2(g,m))
        g.notj[v] = 1
        g.mu2 = g.mu2 * g.notj[v]
        g.mu2 = self.relu(self.theta_v2(self.embeddingMachinemu2(g,m)))



        if batch_size > 1:
            Q = torch.FloatTensor([]).to(device)
            for b in range(batch_size):
                mu_graph = torch.sum(g.mu[(g.batch == b).nonzero().squeeze()], dim=0)
                wgraph = mu_graph

                fis = self.relu(self.embedfinal(g,e[0])[v] + e[1])
                embedmus = self.theta7(fis)
                q = wgraph + embedmus
                Q = torch.cat([Q, q])

        else:
            if len(c2) == 0:
                c_last = c1[v]
            else:
                c_last = max(c1[v], c2[v])

            c_last = c_last+e[1]
            #print(c_last , "Clast")
            v3 = self.theta_v3(g.mu2)[v]
            #print(v3, "V3")
            if Train:
                embedmus = c_last
            else:
                embedmus = v3 + c_last
            q = embedmus
            Q = q




        return Q


