# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_sparse import SparseTensor, matmul
# from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing
# from torch_geometric.nn.conv.gcn_conv import gcn_norm
# import numpy as np
# import scipy.sparse
# from tqdm import tqdm
# import copy

from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_sparse import masked_select_nnz
from torch import Tensor
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter as Param
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse
from tqdm import tqdm
from torch.nn.parameter import Parameter
import torch.nn.init as init
import math
import sys
import copy
from typing import Union, Tuple, Optional

from torch.nn.modules.module import Module

class MLPNORM(nn.Module):
    def __init__(self, nnodes, nfeat, nhid, nclass, dropout, alpha, beta, gamma, delta,
                 norm_func_id, norm_layers, orders, orders_func_id, device):
        super(MLPNORM, self).__init__()
        self.fc1 = nn.Linear(nfeat, nhid)
        self.fc2 = nn.Linear(nhid, nclass)
        self.fc3 = nn.Linear(nhid, nhid)
        self.fc4 = nn.Linear(nnodes, nhid)
        # self.bn1 = nn.BatchNorm1d(nhid)
        # self.bn2 = nn.BatchNorm1d(nhid)
        self.nclass = nclass
        self.dropout = dropout
        self.alpha = torch.tensor(alpha).to(device)
        self.beta = torch.tensor(beta).to(device)
        self.gamma = torch.tensor(gamma).to(device)
        self.delta = torch.tensor(delta).to(device)
        self.norm_layers = norm_layers
        self.orders = orders
        self.device = device
        self.class_eye = torch.eye(self.nclass).to(device)
        self.orders_weight = Parameter(
            (torch.ones(orders, 1) / orders).to(device), requires_grad=True
        )
        self.orders_weight_matrix = Parameter(
            torch.DoubleTensor(nclass, orders).to(device), requires_grad=True
        )
        self.orders_weight_matrix2 = Parameter(
            torch.DoubleTensor(orders, orders).to(device), requires_grad=True
        )
        self.diag_weight = Parameter(
            (torch.ones(nclass, 1) / nclass).to(device), requires_grad=True
        )
        if norm_func_id == 1:
            self.norm = self.norm_func1
        else:
            self.norm = self.norm_func2

        if orders_func_id == 1:
            self.order_func = self.order_func1
        elif orders_func_id == 2:
            self.order_func = self.order_func2
        else:
            self.order_func = self.order_func3

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.orders_weight = Parameter(
            (torch.ones(self.orders, 1) / self.orders).to(self.device), requires_grad=True
        )
        init.kaiming_normal_(self.orders_weight_matrix, mode='fan_out')
        init.kaiming_normal_(self.orders_weight_matrix2, mode='fan_out')
        self.diag_weight = Parameter(
            (torch.ones(self.nclass, 1) / self.nclass).to(self.device), requires_grad=True
        )

    def forward(self, x, adj):
        # xd = F.dropout(x, self.dropout, training=self.training)
        # adjd = F.dropout(adj, self.dropout, training=self.training)
        xX = self.fc1(x)
        # x = self.bn1(x)
        xA = self.fc4(adj)
        x = F.relu(self.delta * xX + (1-self.delta) * xA)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc3(x))
        # x = self.bn2(x)
        feat = F.dropout(x, self.dropout, training=self.training)
        x = self.fc2(feat)
        h0 = x
        for _ in range(self.norm_layers):
            x = self.norm(x, h0, adj)
        return feat,x

    def norm_func1(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1.0 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = coe1 * coe * x - coe1 * coe * coe * torch.mm(x, res)
        tmp = torch.mm(torch.transpose(x, 0, 1), res)
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def norm_func2(self, x, h0, adj):
        coe = 1.0 / (self.alpha + self.beta)
        coe1 = 1 - self.gamma
        coe2 = 1.0 / coe1
        res = torch.mm(torch.transpose(x, 0, 1), x)
        inv = torch.inverse(coe2 * coe2 * self.class_eye + coe * res)
        res = torch.mm(inv, res)
        res = (coe1 * coe * x -
               coe1 * coe * coe * torch.mm(x, res)) * self.diag_weight.t()
        tmp = self.diag_weight * (torch.mm(torch.transpose(x, 0, 1), res))
        sum_orders = self.order_func(x, res, adj)
        res = coe1 * torch.mm(x, tmp) + self.beta * sum_orders - \
            self.gamma * coe1 * torch.mm(h0, tmp) + self.gamma * h0
        return res

    def order_func1(self, x, res, adj):
        tmp_orders = res
        sum_orders = tmp_orders
        for _ in range(self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders
        return sum_orders

    def order_func2(self, x, res, adj):
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = tmp_orders * self.orders_weight[0]
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + tmp_orders * self.orders_weight[i]
        return sum_orders

    def order_func3(self, x, res, adj):
        orders_para = torch.mm(torch.relu(torch.mm(x, self.orders_weight_matrix)),
                               self.orders_weight_matrix2)
        orders_para = torch.transpose(orders_para, 0, 1)
        # tmp_orders = torch.sparse.spmm(adj, res)
        tmp_orders = adj.matmul(res)
        sum_orders = orders_para[0].unsqueeze(1) * tmp_orders
        for i in range(1, self.orders):
            # tmp_orders = torch.sparse.spmm(adj, tmp_orders)
            tmp_orders = adj.matmul(tmp_orders)
            sum_orders = sum_orders + orders_para[i].unsqueeze(1) * tmp_orders
        return sum_orders


class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def update_moving_average(target_ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = target_ema_updater.update_average(old_weight, up_weight)


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


class DSSL(torch.nn.Module):
    def __init__(self, encoder, hidden_channels,dataset, device, cluster_num, alpha,gamma,tao,beta,moving_average_decay,class_size):
        super(DSSL, self).__init__()
        self.dataset = dataset
        self.device = device
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.gamma = gamma
        self.tao = tao
        self.beta=beta
        self.class_size =  class_size
        self.inner_dropout = True
        self.inner_activation = True
        self.online_encoder=encoder
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay)

        ##### class layer
        self.class_layer = nn.Linear(self.class_size, hidden_channels)

        self.mlp_inference = MLP_DSSL(hidden_channels, hidden_channels, cluster_num, 2)
        self.mlp_predictor = MLP_DSSL(hidden_channels, hidden_channels, hidden_channels, 2)
        self.clusters = nn.Parameter(torch.nn.init.normal_(torch.Tensor(hidden_channels, cluster_num)))
        self.mlp_predictor2 = MLP_DSSL(cluster_num, hidden_channels, hidden_channels, 1)

        self.Embedding_mlp = True
        self.inference_mlp = True


    def reset_parameters(self):
        self.class_layer.reset_parameters()
        self.mlp_inference.reset_parameters()
        self.mlp_predictor.reset_parameters()
        self.mlp_predictor2.reset_parameters()
        self.online_encoder.reset_parameters()
        self.target_encoder = copy.deepcopy(self.online_encoder)


    def update_moving_average(self):
        #assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater,self.target_encoder, self.online_encoder)


    def forward(self, embedding,neighbor_embedding):
        # expand node embedding
        embedding_more = embedding[:, None, :]
        embedding_expand = embedding_more.expand(-1, neighbor_embedding.shape[1], -1)
        print(0)
        # print(embedding_expand.size())
        # loss
        if self.inference_mlp == True:
            k, k_node, entropy_loss = self.inference_network(embedding_expand, neighbor_embedding)
        else:
            k, k_node, entropy_loss = self.inference_network2(embedding_expand, neighbor_embedding)
        print(1)
        # if self.Embedding_mlp == True:
        #     main_loss = self.generative_network(embedding_expand, k, neighbor_embedding)
        # else:
        #     main_loss = self.generative_network2(embedding_expand, k, neighbor_embedding)
        context_loss = self.context_network(embedding,k_node)
        print(2)
        return context_loss, k_node
        # return main_loss, context_loss, entropy_loss,k_node
        

    # N × K, N × 10 * K, k is N*32
    def inference_network(self, embedding_expand, neighbor_embedding):
        # get k
        cat_embedding = embedding_expand*neighbor_embedding
        # print(cat_embedding.size())
        # cat_embedding=torch.cat((embedding_expand,neighbor_embedding))
        k = F.softmax(self.mlp_inference(cat_embedding), dim=2)
        k_node = k.mean(dim=1) #to get P(k|x)
        negative_entropy = k_node * torch.log(k_node+1e-10)
        #minimize negative entropy
        entropy_loss = negative_entropy.sum(-1).mean()
        return k, k_node, entropy_loss

    def inference_network2(self, embedding_expand, neighbor_embedding):
        # get k
        cat_embedding = embedding_expand*neighbor_embedding
        k  = torch.matmul(cat_embedding, self.clusters)
        #print(k)
        k_node = k.mean(dim=1) #to get P(k|x)
        negative_entropy = k_node * torch.log(F.softmax(k_node, dim=1)+1e-10)
        #minimize negative entropy
        #print(negative_entropy)
        entropy_loss = negative_entropy.sum(-1).mean()

        return k, k_node, entropy_loss


    def generative_network(self, embedding_expand, k, neighbor_embedding):
        # 
        # re-parameterization trick
        gumbel_k = F.gumbel_softmax(k, hard=False)
        central=self.mlp_predictor(embedding_expand)+ self.beta*self.mlp_predictor2(gumbel_k)
        neighbor=neighbor_embedding
        loss= loss_fn(central, neighbor.detach()).mean()

        return loss

    def generative_network2(self, embedding_expand, k, neighbor_embedding):
        # em algorithm (to do)
        # re-parameterization trick
        gumbel_k = F.gumbel_softmax(k, hard=True)
        central=(embedding_expand)+ self.beta*self.mlp_predictor2(gumbel_k)
        neighbor=neighbor_embedding
        loss= loss_fn(central, neighbor).mean()
        return loss

    def context_network(self, embedding, k_node):
        # self.clusters.data = F.normalize(self.clusters.data, dim=-1, p=2)
        kprior = torch.matmul(embedding, self.clusters)
        kprior = F.softmax(kprior/self.tao, dim=1)

        context_loss = k_node *torch.log(kprior+1e-10)
        context_loss = - 1.0 * context_loss.sum(-1).mean()
        return context_loss


    def update_cluster(self, new_center,batch_sum):
        with torch.no_grad():
            out_ids = torch.arange(self.cluster_num).to(self.device)
            out_ids = out_ids.long()
            self.clusters.index_copy_(1, out_ids, new_center)
            self.clusters.data=torch.mul(self.clusters.data.T ,1/ (batch_sum+1)).T

class MLP_DSSL(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.0):
        super(MLP_DSSL, self).__init__()
        self.linear1 = nn.Linear(in_channels, hidden_channels)
        self.linear2 = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, data):
        x = data
        x = F.relu(self.linear1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.linear2(x)
        return x

class LINKX_Pro(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, num_prototypes_per_class, enable_prot=True, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX_Pro, self).__init__()	
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout
        self.enable_prot = enable_prot
        self.num_prototypes_per_class = num_prototypes_per_class

        #######prototype
        self.enable_prot = self.enable_prot
        self.epsilon = 1e-4
        self.prototype_shape = (out_channels * num_prototypes_per_class, hidden_channels)
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        self.num_prototypes = self.prototype_shape[0]
        self.last_layer = nn.Linear(self.num_prototypes, out_channels,
                                    bias=False)  # do not use bias
        assert (self.num_prototypes % out_channels == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, out_channels)
        # print(self.prototype_class_identity.size())
        # print(self.num_prototypes)
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // self.num_prototypes_per_class] = 1
        # initialize the last layer
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()
        self.last_layer.reset_parameters()

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def prototype_distances(self, x):
        xp = torch.mm(x, torch.t(self.prototype_vectors))
        distance = -2 * xp + torch.sum(x ** 2, dim=1, keepdim=True) + torch.t(
            torch.sum(self.prototype_vectors ** 2, dim=1, keepdim=True))
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance

    def prototype_subgraph_distances(self, x, prototype):
        distance = torch.norm(x - prototype, p=2, dim=1, keepdim=True) ** 2
        similarity = torch.log((distance + 1) / (distance + self.epsilon))
        return similarity, distance


    def forward(self, data):	
        m = data.graph['num_nodes']	
        feat_dim = data.graph['node_feat']	
        row, col = data.graph['edge_index']	
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()

        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(data.graph['node_feat'], input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)

        if self.enable_prot:
            prototype_activations, min_distances = self.prototype_distances(x)
            logits = self.last_layer(prototype_activations)
            # probs = self.Softmax(logits)###原来有log_softmax
            return logits, min_distances
        else:
             x = self.mlp_final(x, input_tensor=True)
             return x


class LINKX(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()	
        self.mlpA = MLP(num_nodes, hidden_channels, hidden_channels, init_layers_A, dropout=0)
        self.mlpX = MLP(in_channels, hidden_channels, hidden_channels, init_layers_X, dropout=0)
        self.W = nn.Linear(2*hidden_channels, hidden_channels)
        self.mlp_final = MLP(hidden_channels, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels
        self.num_nodes = num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	

    def forward(self, data):	
        m = data.graph['num_nodes']	
        feat_dim = data.graph['node_feat']	
        row, col = data.graph['edge_index']	
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()

        _,xA = self.mlpA(A, input_tensor=True)
        _,xX = self.mlpX(data.graph['node_feat'], input_tensor=True)


        ############# temp = torch.mm(y.t(), z)
        ############# temp = torch.mm(y, temp)
        ############## temp = torch.sigmoid(temp)

#       y: Tensor = torch.rand(m, feat_dim).to(A.device)
#       feat = data.graph['node_feat']
#       z:  Tensor = feat
#       lap = xA
#       
        # feat = F.dropout(feat, self.dropout, training=self.training)
        # temp = torch.mm(self.w1, z.t())
        # temp = torch.mm(z, temp)
        # temp = torch.sigmoid(temp)
        # temp = torch.mm(temp, y)
        # temp = (self.lambda_2 / self.lambda_1) * temp
        # y_n = torch.spmm(lap, y) - temp
        # z_n = feat - self.lambda_2 * xX


        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        feat,logits = self.mlp_final(x, input_tensor=True)
        return  feat,logits

class LINK(nn.Module):
    """ logistic regression on adjacency matrix """
    
    def __init__(self, num_nodes, out_channels):
        super(LINK, self).__init__()
        self.W = nn.Linear(num_nodes, out_channels)
        self.num_nodes = num_nodes

    def reset_parameters(self):
        self.W.reset_parameters()
        
    def forward(self, data):
        N = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()
        logits = self.W(A)
        return logits

class LINK_multi(nn.Module):
    """ logistic regression on adjacency matrix """
    
    def __init__(self, num_nodes, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(LINK_multi, self).__init__()
        self.num_nodes = num_nodes

        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(num_nodes, out_channels))
        else:
            self.lins.append(nn.Linear(num_nodes, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        
    def forward(self, data):
        N = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        if isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            row = row-row.min() # for sampling
            A = SparseTensor(row=row, col=col, sparse_sizes=(N, self.num_nodes)).to_torch_sparse_coo_tensor()
        elif isinstance(edge_index, SparseTensor):
            A = edge_index.to_torch_sparse_coo_tensor()

        for i, lin in enumerate(self.lins[:-1]):
            A = lin(A)
            A = F.relu(A, inplace=True)
            A = self.bns[i](A)
            A = F.dropout(A, p=self.dropout, training=self.training)
        logits = self.lins[-1](A)
        return A, logits 

class LINK_Concat(nn.Module):	
    """ concate A and X as joint embeddings i.e. MLP([A;X])"""

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_nodes, dropout=.5, cache=True):	
        super(LINK_Concat, self).__init__()	
        self.mlp = MLP(in_channels + num_nodes, hidden_channels, out_channels, num_layers, dropout=dropout)
        self.in_channels = in_channels	
        self.cache = cache
        self.x = None

    def reset_parameters(self):	
        self.mlp.reset_parameters()	

    def forward(self, data):	
        if (not self.cache) or (not isinstance(self.x, torch.Tensor)):
                N = data.graph['num_nodes']	
                feat_dim = data.graph['node_feat']	
                row, col = data.graph['edge_index']	
                col = col + self.in_channels	
                feat_nz = data.graph['node_feat'].nonzero(as_tuple=True)	
                feat_row, feat_col = feat_nz	
                full_row = torch.cat((feat_row, row))	
                full_col = torch.cat((feat_col, col))	
                value = data.graph['node_feat'][feat_nz]	
                full_value = torch.cat((value, 	
                                torch.ones(row.shape[0], device=value.device)))	
                x = SparseTensor(row=full_row, col=full_col,	
                         sparse_sizes=(N, N+self.in_channels)	
                            ).to_torch_sparse_coo_tensor()	
                if self.cache:
                    self.x = x
        else:
                x = self.x
        logits = self.mlp(x, input_tensor=True)
        return logits


class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.lins[-1](x)
        # return logits
        return x,logits
    
class SGC(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ takes 'hops' power of the normalized adjacency"""
        super(SGC, self).__init__()
        self.conv = SGConv(in_channels, out_channels, hops, cached=True) 

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        x = self.conv(x, edge_index)
        return x


class SGCMem(nn.Module):
    def __init__(self, in_channels, out_channels, hops):
        """ lower memory version (if out_channels < in_channels)
        takes weight multiplication first, then propagate
        takes hops power of the normalized adjacency
        """
        super(SGCMem, self).__init__()

        self.lin = nn.Linear(in_channels, out_channels)
        self.hops = hops

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = data.graph['node_feat']
        x = self.lin(x)
        n = data.graph['num_nodes']
        edge_weight=None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False,
                 dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index

        for _ in range(self.hops):
            x = matmul(adj_t, x)
        
        return x


class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, use_bn=True):
        super(GCN, self).__init__()

        cached = False
        add_self_loops = True
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, out_channels, cached=cached, normalize=not save_mem, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            if self.use_bn:
                x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, sampling=False, add_self_loops=True):
        super(GAT, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True, add_self_loops=add_self_loops) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, out_channels, heads=heads, concat=False, add_self_loops=add_self_loops))

        self.dropout = dropout
        self.activation = F.elu 
        self.sampling = sampling
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, data, adjs=None, x_batch=None):
        if not self.sampling:
            x = data.graph['node_feat']
            for i, conv in enumerate(self.convs[:-1]):
                x = conv(x, data.graph['edge_index'])
                x = self.bns[i](x)
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, data.graph['edge_index'])
        else:
            x = x_batch
            for i, (edge_index, _, size) in enumerate(adjs):
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


    def inference(self, data, subgraph_loader):
        x_all = data.graph['node_feat'] 
        pbar = tqdm(total=x_all.size(0) * self.num_layers)
        pbar.set_description('Evaluating')

        total_edges = 0
        device = x_all.device
        for i in range(self.num_layers):
            xs = []
            for batch_size, n_id, adj in subgraph_loader:
                edge_index, _, size = adj.to(device)
                total_edges += edge_index.size(1)
                x = x_all[n_id].to(device)
                x_target = x[:size[1]]
                x = self.convs[i]((x, x_target), edge_index)
                if i != self.num_layers - 1:
                    x = self.bns[i](x)
                    x = self.activation(x)
                xs.append(x.cpu())

                pbar.update(batch_size)

            x_all = torch.cat(xs, dim=0)

        pbar.close()

        return x_all

class MultiLP(nn.Module):
    """ label propagation, with possibly multiple hops of the adjacency """
    
    def __init__(self, out_channels, alpha, hops, num_iters=50, mult_bin=False):
        super(MultiLP, self).__init__()
        self.out_channels = out_channels
        self.alpha = alpha
        self.hops = hops
        self.num_iters = num_iters
        self.mult_bin = mult_bin # handle multiple binary tasks
        
    def forward(self, data, train_idx):
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight=None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False)
            row, col = edge_index
            # transposed if directed
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False)
            edge_weight=None
            adj_t = edge_index

        y = torch.zeros((n, self.out_channels)).to(adj_t.device())
        if data.label.shape[1] == 1:
            # make one hot
            y[train_idx] = F.one_hot(data.label[train_idx], self.out_channels).squeeze(1).to(y)
        elif self.mult_bin:
            y = torch.zeros((n, 2*self.out_channels)).to(adj_t.device())
            for task in range(data.label.shape[1]):
                y[train_idx, 2*task:2*task+2] = F.one_hot(data.label[train_idx, task], 2).to(y)
        else:
            y[train_idx] = data.label[train_idx].to(y.dtype)
        result = y.clone()
        for _ in range(self.num_iters):
            for _ in range(self.hops):
                result = matmul(adj_t, result)
            result *= self.alpha
            result += (1-self.alpha)*y

        if self.mult_bin:
            output = torch.zeros((n, self.out_channels)).to(result.device)
            for task in range(data.label.shape[1]):
                output[:, task] = result[:, 2*task+1]
            result = output

        return result


class MixHopLayer(nn.Module):
    """ Our MixHop layer """
    def __init__(self, in_channels, out_channels, hops=2):
        super(MixHopLayer, self).__init__()
        self.hops = hops
        self.lins = nn.ModuleList()
        for hop in range(self.hops+1):
            lin = nn.Linear(in_channels, out_channels)
            self.lins.append(lin)

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x, adj_t):
        xs = [self.lins[0](x) ]
        for j in range(1,self.hops+1):
            # less runtime efficient but usually more memory efficient to mult weight matrix first
            x_j = self.lins[j](x)
            for hop in range(j):
                x_j = matmul(adj_t, x_j)
            xs += [x_j]
        return torch.cat(xs, dim=1)

class MixHop(nn.Module):
    """ our implementation of MixHop
    some assumptions: the powers of the adjacency are [0, 1, ..., hops],
        with every power in between
    each concatenated layer has the same dimension --- hidden_channels
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, hops=2):
        super(MixHop, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(MixHopLayer(in_channels, hidden_channels, hops=hops))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))
        for _ in range(num_layers - 2):
            self.convs.append(
                MixHopLayer(hidden_channels*(hops+1), hidden_channels, hops=hops))
            self.bns.append(nn.BatchNorm1d(hidden_channels*(hops+1)))

        self.convs.append(
            MixHopLayer(hidden_channels*(hops+1), out_channels, hops=hops))

        # note: uses linear projection instead of paper's attention output
        self.final_project = nn.Linear(out_channels*(hops+1), out_channels)

        self.dropout = dropout
        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False,
                 dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False,
                dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)

        x = self.final_project(x)
        return x

class GCNJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, save_mem=False, jk_type='max'):
        super(GCNJK, self).__init__()

        cached = False
        self.convs = nn.ModuleList()
        self.convs.append(
            GCNConv(in_channels, hidden_channels, cached=cached, normalize=not save_mem))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.convs.append(
            GCNConv(hidden_channels, hidden_channels, cached=cached, normalize=not save_mem))

        self.dropout = dropout
        self.activation = F.relu
        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels * num_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x

class GATJK(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2,
                 dropout=0.5, heads=2, jk_type='max'):
        super(GATJK, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=heads, concat=True))

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*heads))
        for _ in range(num_layers - 2):

            self.convs.append(
                    GATConv(hidden_channels*heads, hidden_channels, heads=heads, concat=True) ) 
            self.bns.append(nn.BatchNorm1d(hidden_channels*heads))

        self.convs.append(
            GATConv(hidden_channels*heads, hidden_channels, heads=heads))

        self.dropout = dropout
        self.activation = F.elu # note: uses elu

        self.jump = JumpingKnowledge(jk_type, channels=hidden_channels*heads, num_layers=1)
        if jk_type == 'cat':
            self.final_project = nn.Linear(hidden_channels*heads*num_layers, out_channels)
        else: # max or lstm
            self.final_project = nn.Linear(hidden_channels*heads, out_channels)


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.jump.reset_parameters()
        self.final_project.reset_parameters()


    def forward(self, data):
        x = data.graph['node_feat']
        xs = []
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, data.graph['edge_index'])
            x = self.bns[i](x)
            x = self.activation(x)
            xs.append(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, data.graph['edge_index'])
        xs.append(x)
        x = self.jump(xs)
        x = self.final_project(x)
        return x

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)


class H2GCN(nn.Module):
    """ our implementation """
    def __init__(self, in_channels, hidden_channels, out_channels, edge_index, num_nodes,
                    num_layers=2, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()

        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)


        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)
        self.final_project = nn.Linear(last_dim, out_channels)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(edge_index.device)
        self.adj_t2 = adj_t2.to(edge_index.device)



    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(data)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_project(x)
        return x


class APPNP_Net(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dprate=.0, dropout=.5, K=10, alpha=.1, num_layers=3):
        super(APPNP_Net, self).__init__()
        
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = APPNP(K, alpha)

        self.dprate = dprate
        self.dropout = dropout
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x





class GPR_prop(MessagePassing):
    '''
    GPRGNN, from original repo https://github.com/jianhao2016/GPRGNN
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        if isinstance(edge_index, torch.Tensor):
            edge_index, norm = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)
            norm = None

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

class GPRGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, Init='Random', dprate=.0, dropout=.5, K=10, alpha=.1, Gamma=None, num_layers=3):
        super(GPRGNN, self).__init__()
        
        self.mlp = MLP(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.prop1 = GPR_prop(K, alpha, Init, Gamma)

        self.Init = Init
        self.dprate = dprate
        self.dropout = dropout
        
    def reset_parameters(self):
        self.mlp.reset_parameters()
        self.prop1.reset_parameters()

    def forward(self, data):
        edge_index = data.graph['edge_index']
        x = self.mlp(data)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return x
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return x


class GCNII(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, alpha, theta, shared_weights=True, dropout=0.5):
        super(GCNII, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        self.bns = nn.ModuleList()
        self.convs = nn.ModuleList()
        for layer in range(num_layers):
            self.convs.append(
                GCN2Conv(hidden_channels, alpha, theta, layer + 1,
                         shared_weights, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        self.dropout = dropout
        
    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x = data.graph['node_feat']
        n = data.graph['num_nodes']
        edge_index = data.graph['edge_index']
        edge_weight = None
        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False, dtype=x.dtype)
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False, dtype=x.dtype)
            edge_weight=None
            adj_t = edge_index
        
        x = F.dropout(x, self.dropout, training=self.training)
        x = x_0 = self.lins[0](x).relu()
        
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, self.dropout, training=self.training)
            x = conv(x, x_0, adj_t)
            x = self.bns[i](x)
            x = x.relu()

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lins[1](x)
        return x
