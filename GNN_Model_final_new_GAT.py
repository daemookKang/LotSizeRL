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
    def __init__(self, num_feats,heads,concat,negative_slope,dropout,add_self_loops,bias):
        super(Net_MLP, self).__init__()
        self.num_feats = num_feats
        self.embeddingPrecedence = EmbeddingPrecedence(num_feats)
        self.embeddingMachinemu2 = EmbeddingMachinemu2(num_feats)
        self.embeddingMachine2 = EmbeddingMachine2(num_feats)
        self.embedC = EmbedC(num_feats)
        self.theta_v = nn.Linear(1, num_feats, bias=False).to(device)
        self.theta_v2 = nn.Linear(num_feats, num_feats, bias=False).to(device)
        self.theta_v3 = nn.Linear(num_feats*heads, 1, bias=False).to(device)
        self.theta_v4 = nn.Linear(num_feats , 1, bias=False).to(device)

        self.in_channels = num_feats
        self.out_channels = num_feats
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.bias = bias

        self.gat = GATConv(self.in_channels, self.out_channels, self.heads, self.concat,
                                 self.negative_slope, self.dropout, self.add_self_loops, self.bias).to(device)




        self.relu = nn.ReLU().to(device)


    def forward(self, g,v, e,m,batch_size=1, Train = True,only_clast=False):
        c1  = self.embedC(g,e[0])
        g.notj[v] = 0
        if len(g.edge_index_P) == 0:
            c2 = []
        else:
            c2 = self.embeddingPrecedence(g)



        g.mu2 = self.theta_v(self.embeddingMachine2(g,m))
        g.notj[v] = 1
        g.mu2 = g.mu2 * g.notj[v]
        m_edge = g.index_m[m - 1].t()
        g.mu2 = self.gat(g.mu2,m_edge)
        #g.mu2 = self.relu(self.theta_v2(self.embeddingMachinemu2(g,m)))



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
            # v4 = self.theta_v4(self.relu(v3))

            #print(v3, "V3")
            if only_clast:
                embedmus = c_last

            elif Train:
                embedmus = v3
            else:
                print(c_last, "Clast")
                print(v3, "V3")
                embedmus = v3 + c_last
            q = embedmus
            Q = q




        return Q


class GATConv(MessagePassing):
    _alpha: OptTensor
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        super(GATConv, self).__init__(aggr='add', node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * out_channels, False)

        self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        if isinstance(x, Tensor):

            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)

        assert x_l is not None
        assert alpha_l is not None

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = x_l.size(0)
                if x_r is not None:
                    num_nodes = min(num_nodes, x_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r),
                             alpha=(alpha_l, alpha_r), size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:

        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
