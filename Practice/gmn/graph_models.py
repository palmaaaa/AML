# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MetaLayer
from torch_geometric.nn.pool import max_pool_x, avg_pool_x, global_max_pool, global_mean_pool
from torch_scatter import scatter

class EdgeModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True):
        super().__init__()
        if activation:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.edge_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))

    def forward(self, src, dest, edge_attr, u, batch):
        # **IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # src, dest: [E, F_x], where E is the number of edges. src is the source node features and dest is the destination node features of each edge.
        # edge_attr: [E, F_e]
        # u: [B, F_u], where B is the number of graphs.
        # batch: only here it will have shape [E] with max entry B - 1, because here it indicates the graph index for each edge.

        x = [dest, src, edge_attr, u[batch]]
        x = torch.cat(x, dim=-1)
        return self.edge_mlp(x)


class NodeModel(nn.Module):
    def __init__(self, in_dim_mlp1, in_dim_mlp2, out_dim, activation=True, reduce='sum'):
        super().__init__()
        self.reduce = reduce
        if activation:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim_mlp1, out_dim), nn.ReLU())
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim_mlp2, out_dim), nn.ReLU())
        else:
            self.node_mlp_1 = nn.Sequential(nn.Linear(in_dim_mlp1, out_dim))
            self.node_mlp_2 = nn.Sequential(nn.Linear(in_dim_mlp2, out_dim))

    def forward(self, x, edge_index, edge_attr, u, batch):
        # **IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.

        col, row = edge_index 
        edge_messages = self.node_mlp_1(torch.cat([x[row], x[col], edge_attr, u[batch[row]]], dim=1))  

        aggregated_messages = scatter(edge_messages, row, dim=0, reduce=self.reduce, dim_size=x.size(0)) 

        global_info = u[batch] 
        updated_node_features = self.node_mlp_2(torch.cat([x, aggregated_messages, global_info], dim=1))
        return updated_node_features


class GlobalModel(nn.Module):
    def __init__(self, in_dim, out_dim, activation=True, reduce='sum'):
        super().__init__()
        if activation:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
        else:
            self.global_mlp = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.reduce = reduce

    def forward(self, x, edge_index, edge_attr, u, batch):
        #**IMPORTANT: YOU ARE NOT ALLOWED TO USE FOR LOOPS!**
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        # u: [B, F_u]
        # batch: [N] with max entry B - 1.


        node_sum = scatter(x, batch, dim=0, reduce=self.reduce) 

        edge_sum = scatter(edge_attr, batch[edge_index[0]], dim=0, reduce=self.reduce) 

        global_input = torch.cat([node_sum, edge_sum, u], dim=1) 

        updated_u = self.global_mlp(global_input)

        return updated_u


class MPNN(nn.Module):

    def __init__(self, node_in_dim, edge_in_dim, global_in_dim, hidden_dim, node_out_dim, edge_out_dim, global_out_dim, num_layers,
                use_bn=True, dropout=0.0, reduce='sum'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.node_norms = nn.ModuleList()
        self.edge_norms = nn.ModuleList()
        self.global_norms = nn.ModuleList()
        self.use_bn = use_bn
        self.dropout = dropout
        self.reduce = reduce

        assert num_layers >= 2

        '''
        Instantiate the first layer models with correct parameters below
        '''

        edge_model = EdgeModel(node_in_dim * 2 + edge_in_dim + global_in_dim, hidden_dim)
        node_model = NodeModel(node_in_dim * 2 + hidden_dim + global_in_dim, node_in_dim + hidden_dim + global_in_dim, hidden_dim)
        global_model = GlobalModel(hidden_dim + hidden_dim + global_in_dim, hidden_dim)
        self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
        self.node_norms.append(nn.BatchNorm1d(hidden_dim))
        self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
        self.global_norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers-2):
            '''
            Add your code below
            '''
            # add batch norm after each MetaLayer
            edge_model = EdgeModel(hidden_dim * 2 + hidden_dim + hidden_dim, hidden_dim)
            node_model = NodeModel(hidden_dim * 2 + hidden_dim + hidden_dim, hidden_dim + hidden_dim + hidden_dim, hidden_dim)
            global_model = GlobalModel(hidden_dim + hidden_dim + hidden_dim, hidden_dim)
            self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))
            self.node_norms.append(nn.BatchNorm1d(hidden_dim))
            self.edge_norms.append(nn.BatchNorm1d(hidden_dim))
            self.global_norms.append(nn.BatchNorm1d(hidden_dim))


        '''
        Add your code below
        '''
        # last MetaLayer without batch norm and without using activation functions
        edge_model = EdgeModel(hidden_dim * 2 + hidden_dim + hidden_dim, edge_out_dim, activation=False)
        node_model = NodeModel(hidden_dim * 2 + hidden_dim + edge_out_dim, hidden_dim + hidden_dim + node_out_dim, node_out_dim, activation=False)
        global_model = GlobalModel(hidden_dim + edge_out_dim + node_out_dim, global_out_dim, activation=False)
        self.convs.append(MetaLayer(edge_model=edge_model, node_model=node_model, global_model=global_model))


    def forward(self, x, edge_index, edge_attr, u, batch, *args):

        for i, conv in enumerate(self.convs):
            '''
            Add your code below
            '''
            x, edge_attr, u = conv(x, edge_index, edge_attr, u, batch)

            if i != len(self.convs)-1 and self.use_bn:
                '''
                Add your code below this line, but before the dropout
                '''

                x = self.node_norms[i](x)
                edge_attr = self.edge_norms[i](edge_attr)
                u = self.global_norms[i](u)

                x = F.dropout(x, p=self.dropout, training=self.training)
                edge_attr = F.dropout(edge_attr, p=self.dropout, training=self.training)
                u = F.dropout(u, p=self.dropout, training=self.training)

        return x, edge_attr, u