import argparse
import sys
import os
from tqdm import tqdm
import numpy as np
import math
from numpy import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch_geometric.utils import to_undirected
from torch_geometric.data import NeighborSampler
from torch_scatter import scatter
from torch.optim.lr_scheduler import LambdaLR
from os import path
from logger import Logger, SimpleLogger
from dataset import load_nc_dataset
from correct_smooth import double_correlation_autoscale, double_correlation_fixed
from data_utils import normalize, gen_normalized_adjs, evaluate, evaluate_pro, evaluate_mlpnorm_hom,eval_acc, eval_rocauc, to_sparse_tensor, load_fixed_splits
from parse import parse_method, parser_add_main_args
import faulthandler; faulthandler.enable()

from torch_scatter import scatter_mean
from torch_geometric.utils import remove_self_loops
from torch_sparse import SparseTensor
# NOTE: for consistent data splits, see data_utils.rand_train_test_idx

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)

parser.add_argument('--infer_env_lr', type=float,default=0.001)
parser.add_argument("--z_class_num", type=int, default= 4)#####  environment number
parser.add_argument("--hidden_dim_infer", type=int, default= 16)
parser.add_argument("--penalty_anneal_iters", type=int, default= 5)
parser.add_argument('--penalty_weight', type=float, default=10000.0)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--scheduler', type=int, default=0)
parser.add_argument('--steps', type=int, default=501)
parser.add_argument('--distance_method', type=str, default='norm2')

args = parser.parse_args()
print(args)

random.seed(args.seeds)
np.random.seed(args.seeds)
torch.manual_seed(args.seeds)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seeds)

device = torch.device(args.device)
if args.cpu:
    device = torch.device('cpu')
hom_list = []

# if args.method == 'mlpnorm':
#     torch.set_default_dtype(torch.float64)

### Load and preprocess data ###
dataset = load_nc_dataset(args.dataset, args.sub_dataset)
if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

if args.rand_split or args.dataset in ['ogbn-proteins', 'wiki']:
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                for _ in range(args.runs)]
else:
    split_idx_lst = load_fixed_splits(args.dataset, args.sub_dataset)

if args.dataset == 'ogbn-proteins':
    if args.method == 'mlp' or args.method == 'cs':
        dataset.graph['node_feat'] = scatter(dataset.graph['edge_feat'], dataset.graph['edge_index'][0],
            dim=0, dim_size=dataset.graph['num_nodes'], reduce='mean')
    else:
        dataset.graph['edge_index'] = to_sparse_tensor(dataset.graph['edge_index'],
            dataset.graph['edge_feat'], dataset.graph['num_nodes'])
        dataset.graph['node_feat'] = dataset.graph['edge_index'].mean(dim=1)
        dataset.graph['edge_index'].set_value_(None)
    dataset.graph['edge_feat'] = None

n = dataset.graph['num_nodes']
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

# whether or not to symmetrize matters a lot!! pay attention to this
# e.g. directed edges are temporally useful in arxiv-year,
# so we usually do not symmetrize, but for label prop symmetrizing helps
if not args.directed and args.dataset != 'ogbn-proteins':
    dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(
        device), dataset.graph['node_feat'].to(device)

x=None
adj=None
if args.method == 'mlpnorm':
    x = dataset.graph['node_feat']
    edge_index = dataset.graph['edge_index']
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(
        dataset.graph['num_nodes'], dataset.graph['num_nodes'])).to_torch_sparse_coo_tensor()
    # adj = adj.to_dense()
    x = x.to(device)
    adj = adj.to(device)
    x = x.to(torch.float)
    adj = adj.to(torch.float)

train_loader, subgraph_loader = None, None

print(f"num nodes {n} | num classes {c} | num node feats {d}")

### Load method ###
model = parse_method(args, dataset, n, c, d, device)

#######################################################################################Zin
class InferEnvMultiClass(nn.Module):
    def __init__(self, args, z_dim, class_num):
        super(InferEnvMultiClass, self).__init__()
        self.lin1 = nn.Linear(z_dim, args.hidden_dim_infer)
        self.lin2 = nn.Linear(args.hidden_dim_infer, class_num)
        for lin in [self.lin1, self.lin2]:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        self._main = nn.Sequential(
            self.lin1, nn.ReLU(True), self.lin2, nn.Softmax(dim=1))
    def forward(self, input):
        out = self._main(input)
        return out

class Infer_Irmv1_Multi_Class:
    
    def __init__(self, args):
        infer_env = InferEnvMultiClass(args, z_dim=1, class_num=args.z_class_num).cuda(args.device)
        self.optimizer_infer_env = torch.optim.Adam(infer_env.parameters(), lr=args.infer_env_lr)
        self.args = args
        self.infer_env = infer_env

    def __call__(self, dataset, train_idx, hom_train, epoch, model, x, adj, scale=None, **kwargs):
        train_z = hom_train
        normed_z = (train_z.float() - train_z.float().mean())/train_z.float().std()
        # print(type(x))
        # print(type(adj))
        if x!=None:
            feat, out = model(x,adj)
        else:
            feat, out = model(dataset)
        out = F.log_softmax(out, dim=1)
        train_logits = scale * out[train_idx]
        train_nll = criterion(train_logits, dataset.label.squeeze(1)[train_idx])
        # print("train_nll:",train_nll.unsqueeze(1).size())
        infered_envs = self.infer_env(normed_z)
        # print("infered_envs:",infered_envs.size())
        train_penalty = 0
        multi_loss = (train_nll.unsqueeze(1) * infered_envs).mean(axis=0) 
        for i in range(multi_loss.shape[0]):
            grad = autograd.grad(
                multi_loss[i],
                [scale],
                create_graph=True)[0]
            train_penalty = train_penalty + grad ** 2
    
        train_nll = train_nll.mean()

        if epoch < self.args.penalty_anneal_iters:
            # gradient ascend on infer_env net
            self.optimizer_infer_env.zero_grad()    
            (-train_penalty).backward(retain_graph=True)
            self.optimizer_infer_env.step()
        return train_nll, train_penalty

class CosineLR(LambdaLR):

    def __init__(self, optimizer, lr, num_epochs, offset=1):
        self.init_lr = lr
        fn = lambda epoch: lr * 0.5 * (1 + np.cos((epoch - offset) / num_epochs * np.pi))
        super().__init__(optimizer, lr_lambda=fn)

    def reset(self, epoch, num_epochs):
        self.__init__(self.optimizer, self.init_lr, num_epochs, offset=epoch)
#######################################################################################################

# using rocauc as the eval function
# def weight_loss(args):
#     if args.weight == 1:
#         if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
#             criterion = nn.BCEWithLogitsLoss(reduction='none')
#             eval_func = eval_rocauc
#         else:
#             criterion = nn.NLLLoss(reduction='none')
#             eval_func = eval_acc
#     else :
#         if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
#             criterion = nn.BCEWithLogitsLoss()
#             eval_func = eval_rocauc
#         else:
#             criterion = nn.NLLLoss()
#             eval_func = eval_acc
#     return criterion, eval_func

if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins','genius'):
    criterion = nn.BCEWithLogitsLoss()
    eval_func = eval_rocauc
else:
    # criterion = nn.NLLLoss()
    criterion = nn.NLLLoss(reduction='none')
    eval_func = eval_acc
logger = Logger(args.runs, args)

if args.method == 'cs':
    cs_logger = SimpleLogger('evaluate params', [], 2)
    model_path = f'{args.dataset}-{args.sub_dataset}' if args.sub_dataset else f'{args.dataset}'
    model_dir = f'models/{model_path}'
    print(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    DAD, AD, DA = gen_normalized_adjs(dataset)

if args.method == 'lp':
    # handles label propagation separately
    for alpha in (.01, .1, .25, .5, .75, .9, .99):
        logger = Logger(args.runs, args)
        for run in range(args.runs):
            split_idx = split_idx_lst[run]
            train_idx = split_idx['train']
            model.alpha = alpha
            out = model(dataset, train_idx)
            result = evaluate(model, dataset, split_idx, eval_func, result=out)
            logger.add_result(run, result[:-1])
            print(f'alpha: {alpha} | Train: {100*result[0]:.2f} ' +
                    f'| Val: {100*result[1]:.2f} | Test: {100*result[2]:.2f}')

        best_val, best_test = logger.print_statistics()
        filename = f'results/{args.dataset}'+f'_{args.seeds}.csv'
        print(f"Saving results to {filename}")
        with open(f"{filename}", 'a+') as write_obj:
            sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
            write_obj.write(f"{args.method}," + f"{sub_dataset}" +
                        f"{best_val.mean():.3f} ± {best_val.std():.3f}," +
                        f"{best_test.mean():.3f} ± {best_test.std():.3f}\n")
    sys.exit()

model.train()
print('MODEL:', model)

def list_txt(path, list=None):
    if list != None:
        file = open(path, 'w')
        file.write(str(list))
        file.close()
        return None
    else:
        file = open(path, 'r')
        rdlist = eval(file.read())
        file.close()
        return rdlist

def spilt_train(train_idx_list, ratio=0.8, shuffle=True):
    n_total = len(train_idx_list)
    offset = int(n_total * ratio)
    if n_total==0 or offset<1:
        return [],train_idx_list
    if shuffle:
        random.shuffle(train_idx_list)
    sublist_1 = train_idx_list[:offset]
    sublist_2 = train_idx_list[offset:]
    return sublist_1

import homophily
def get_idx_list(node_homophily_list,train_idx_list, ratio, tmp):
    length=len(train_idx_list)
    train_hom_list = [0]*length
    for i in range(length):
        train_hom_list[i] = (node_homophily_list[train_idx_list[i]])
    sorted_train_hom_list = sorted(train_hom_list, key = lambda x : float('inf') if (x != x) else x)
    # num = 0
    # i = 0
    # for key in sorted_train_hom_list:
    #     if(key == key):
    #         num = num + key
    #         i = i+1
    # print(num/i)
    # filename_max = f'data/{args.dataset}'+f'_{ratio}_sorted_txt'
    # list_txt(filename_max, sorted_train_hom_list)
    # print(sorted_train_hom_list[0])
    print(round(ratio*length))
    print(round((1-ratio)*length))
    temp_min = sorted_train_hom_list[round(ratio*length)]
    temp_max = sorted_train_hom_list[round((1-ratio)*length)]
    print("temp_min:",temp_min)
    print("temp_max:",temp_max)
    # print(1/0)
    # if(tmp!=0):
    #     temp_max = tmp
    #     temp_min = tmp
    train_min_idx = []
    train_max_idx = []
    # tmp_list = []
    for index in train_idx_list:
        if(node_homophily_list[index]>=temp_max):
            train_max_idx.append(index)
        if(node_homophily_list[index]<temp_min):
            train_min_idx.append(index)
        # if(node_homophily_list[index]==temp_min):
        #     tmp_list.append(index)
    # print(len(train_max_idx))
    # print(len(train_min_idx))
    # print(len(tmp_list))
    # print(1/0)
    return train_max_idx,train_min_idx

def get_renode_weight(args,node_homophily_list,train_idx_list):

    base_w  = args.rn_base_weight
    scale_w = args.rn_scale_weight
    length=len(train_idx_list)

    # train_hom_list = [0]*length
    # for i in range(length):
    #     train_hom_list[i] = (node_homophily_list[train_idx_list[i]])
    # sorted_train_hom_list = sorted(train_hom_list, key = lambda x : float('inf') if (x != x) else x)

    #computing the ReNode Weight
    id2totoro = {}
    num = 0
    for i in train_idx_list:
        if node_homophily_list[i] == node_homophily_list[i]:
            id2totoro[i] = node_homophily_list[i] 
            num = num + 1
    sorted_totoro = sorted(id2totoro.items(),key=lambda x:x[1],reverse=False)
    # print(sorted_totoro)
    id2rank= {sorted_totoro[i][0]:i for i in range(num)}
    rn_weight = []
    for index in train_idx_list:
        if node_homophily_list[index] == node_homophily_list[index]:
            rn_weight.append(base_w + 0.5 * scale_w * (1 + math.cos(id2rank[index]*1.0*math.pi/(num))))
        else:
            rn_weight.append(1)
    # rn_weight = [(base_w + 0.5 * scale_w * (1 + math.cos(x*1.0*math.pi/(length-1)))) for x in totoro_rank]
    rn_weight = torch.from_numpy(np.array(rn_weight)).type(torch.FloatTensor)
    return rn_weight


dataset.graph['edge_index'], dataset.label= dataset.graph['edge_index'].to('cpu'), dataset.label.to('cpu')
node_homophily_list = homophily.node_homophily_edge_idx(dataset.graph['edge_index'], dataset.label, dataset.graph['num_nodes'])
####################
if(args.dataset=="arxiv-year"):
    for index,value in enumerate(node_homophily_list):
        if(value!=value):
            node_homophily_list[index]=1

def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def create_dir(path):
    try:
        os.mkdir(path)
    except:
        print(" fail to create file")
        pass  

create_dir("models")
def get_prototype_embedding(embedding, label, train_idx, c):
    train_idx_list = train_idx.tolist()
    label_list = label.tolist()
    # print("train number:", len(train_idx_list))

    dic={}
    for index in train_idx_list:
        dic.setdefault(label_list[index],[]).append(index)
    
    mean_embedding={}
    sum = 0
    for key,value in dic.items():
        sum = sum + len(value)
        # print(embedding[value])
        mean_embedding[key] = embedding[value].mean(axis =0)
    # print("dic number:", sum)
    # print(len(dic))
    # print(1/0)
    return dic, mean_embedding

def dist(x, edge_index,args):
    # print(x.size())
    # print(edge_index.size())
    # edge_index = remove_self_loops(edge_index)[0]
    # print(edge_index.size())
    # print(1/0)
    src, tgt = edge_index
    # print(src.size())
    # print(tgt.size())
    # print(1/0)
    if args.dataset not in ["fb100","arxiv-year","twitch-gamer"]:
        if args.distance_method == 'cos':
            dist = (x[src] * x[tgt]).sum(dim=-1)
        elif args.distance_method == 'norm2':
            dist = torch.norm(x[src] - x[tgt], p=2, dim=-1)
    else:
        split_size = 10000
        dist = []
        for ei_i in tqdm(edge_index.split(split_size, dim=-1), ncols=70):
            src_i, tgt_i = ei_i
            if args.distance_method == 'cos':                    
                dist.append((x[src_i] * x[tgt_i]).sum(dim=-1))
            elif args.distance_method == 'norm2':
                dist.append(torch.norm(x[src_i] - x[tgt_i], p=2, dim=-1))
        dist = torch.cat(dist, dim=0)
    # print(dist.size())
    # dist = dist.view(-1, 1)
    # print(dist.size())
    # print(tgt)
    local_sim = scatter_mean(dist, src, out=torch.zeros([x.shape[0]], device=x.device))
    # print(local_sim)
    print(local_sim.size())
    print(1/0)
    return local_sim

############## Training loop ##############
for run in range(args.runs):
    split_idx = split_idx_lst[run]
    train_idx_list = split_idx['train'].tolist()
    train_idx = split_idx['train'].to(device)

    ###############test split
    test_filename_max = f'Hom_test/{args.dataset}'+f'_{run}_test_max_txt'
    test_filename_min = f'Hom_test/{args.dataset}'+f'_{run}_test_min_txt'
    if not path.exists(test_filename_max) or not path.exists(test_filename_min):
        print("create test on the hom")
        dataset.graph['edge_index'], dataset.label= dataset.graph['edge_index'].to('cpu'), dataset.label.to('cpu')
        node_homophily_list = homophily.node_homophily_edge_idx(dataset.graph['edge_index'], dataset.label, dataset.graph['num_nodes'])
        split_idx = split_idx_lst[run]
        test_idx_list = split_idx['test'].tolist()
        test_max_idx,test_min_idx = get_idx_list(node_homophily_list,test_idx_list,args.ratio,args.tmp)
        list_txt(test_filename_max, test_max_idx)
        list_txt(test_filename_min, test_min_idx)
    
    test_max_idx = list_txt(test_filename_max)
    test_max_idx = torch.tensor(test_max_idx).to(device)
    test_min_idx = list_txt(test_filename_min)
    test_min_idx = torch.tensor(test_min_idx).to(device)

    ####################################################################################################################################
    # print(max( train_idx_list))
    # filename_max = f'data/{args.dataset}'+f'_{args.ratio}_max_txt'
    # filename_min = f'data/{args.dataset}'+f'_{args.ratio}_min_txt'
    # if not path.exists(filename_max) or not path.exists(filename_min):
    #     filename_max = f'data/{args.dataset}'+f'_{args.ratio}_max_txt'
    #     filename_min = f'data/{args.dataset}'+f'_{args.ratio}_min_txt'
    #     print("create train on the hom")
    #     dataset.graph['edge_index'], dataset.label= dataset.graph['edge_index'].to('cpu'), dataset.label.to('cpu')
    #     node_homophily_list = homophily.node_homophily_edge_idx(dataset.graph['edge_index'], dataset.label, dataset.graph['num_nodes'])
    #     train_max_idx,train_min_idx = get_idx_list(node_homophily_list,train_idx_list,args.ratio,args.tmp)
    #     list_txt(filename_max, train_max_idx)
    #     list_txt(filename_min, train_min_idx)
    # else:
    #     if args.temp == "max":
    #         filename_max = f'data/{args.dataset}'+f'_{args.ratio}_max_txt'
    #         train_max_idx = list_txt(filename_max)
    #     if args.temp == "min":
    #         filename_min = f'data/{args.dataset}'+f'_{args.ratio}_min_txt'
    #         train_min_idx = list_txt(filename_min)

    # if args.temp == "max":
    #     train_idx = torch.tensor(train_max_idx).to(device)
    # if args.temp == "min":
    #     train_idx = torch.tensor(train_min_idx).to(device)
    
    # dataset.graph['edge_index'], dataset.label= dataset.graph['edge_index'].to('cpu'), dataset.label.to('cpu')
    # node_homophily_list = homophily.node_homophily_edge_idx(dataset.graph['edge_index'], dataset.label, dataset.graph['num_nodes'])        
    # rn_weight = get_renode_weight(args,node_homophily_list,train_idx_list)
    # rn_weight = rn_weight.to(device)
    # dataset.graph['edge_index'], dataset.label= dataset.graph['edge_index'].to(device), dataset.label.to(device)
    ####################################################################################################################################
    
    dataset.graph['edge_index'], dataset.label= dataset.graph['edge_index'].to(device), dataset.label.to(device)

    similarity_filename = f'Local_Sim/{args.dataset}'+f'_{args.distance_method}_{run}_txt'
    if not path.exists(similarity_filename):
        print("create similarity on the X")
        src, tgt = dataset.graph['edge_index']
        if args.dataset not in ["fb100","arxiv-year","twitch-gamer"]:
            if args.distance_method == 'cos':
                dist = (dataset.graph['node_feat'][src] * dataset.graph['node_feat'][tgt]).sum(dim=-1)
            elif args.distance_method == 'norm2':
                dist = torch.norm(dataset.graph['node_feat'][src] - dataset.graph['node_feat'][tgt], p=2, dim=-1)
        else:
            split_size = 10000
            dist = []
            for ei_i in tqdm(dataset.graph['edge_index'].split(split_size, dim=-1), ncols=70):
                src_i, tgt_i = ei_i
                if args.distance_method == 'cos':                    
                    dist.append((dataset.graph['node_feat'][src_i] * dataset.graph['node_feat'][tgt_i]).sum(dim=-1))
                elif args.distance_method == 'norm2':
                    dist.append(torch.norm(dataset.graph['node_feat'][src_i] - dataset.graph['node_feat'][tgt_i], p=2, dim=-1))
            dist = torch.cat(dist, dim=0)
        local_sim = scatter_mean(dist, src, out=torch.zeros([dataset.graph['node_feat'].shape[0]], device=dataset.graph['node_feat'].device))
        # print(local_sim.size())
        local_sim_list = local_sim.tolist()
        # print(len(local_sim_list))
        list_txt(similarity_filename, local_sim_list)

    local_sim_list = list_txt(similarity_filename)
    local_sim = torch.tensor(local_sim_list).to(device)

    if args.sampling:
        if args.num_layers == 2:
            sizes = [15, 10]
        elif args.num_layers == 3:
            sizes = [15, 10, 5]
        train_loader = NeighborSampler(dataset.graph['edge_index'], node_idx=train_idx,
                                sizes=sizes, batch_size=1024,
                                shuffle=True, num_workers=12)
        subgraph_loader = NeighborSampler(dataset.graph['edge_index'], node_idx=None, sizes=[-1],
                                        batch_size=4096, shuffle=False,
                                        num_workers=12)

    model.reset_parameters()
    if args.adam:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.SGD:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, nesterov=args.nesterov, momentum=args.momentum)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
    if args.scheduler:
        scheduler = CosineLR(optimizer, args.lr, args.steps)
    best_val = float('-inf')
    # criterion, eval_func = weight_loss(args)
    early_stop_count = 0
    scale =  torch.tensor(1.).cuda(args.device).requires_grad_()
    aux = Infer_Irmv1_Multi_Class(args)


    ### local_sim used for estimating the neighbor pattern
    train_aux = local_sim[train_idx]


    for epoch in range(args.epochs):
        model.train()
        if not args.sampling:
            # out= model(dataset)
            # if args.rocauc or args.dataset in ('yelp-chi', 'twitch-e', 'ogbn-proteins', 'genius'):
            #     if dataset.label.shape[1] == 1:
            #         true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
            #     else:
            #         true_label = dataset.label
            #     loss = criterion(out[train_idx], true_label.squeeze(1)[train_idx].to(torch.float))
            # else:
            #     out = F.log_softmax(out, dim=1)
            #     loss = criterion(out[train_idx], dataset.label.squeeze(1)[train_idx])
            # print(train_hom.unsqueeze(1).size())
            train_nll, train_penalty = aux(dataset, train_idx, train_aux.unsqueeze(1), epoch, model, x, adj, scale)
            weight_norm = torch.tensor(0.).cuda(args.device)
            for w in model.parameters():
                weight_norm = weight_norm +w.norm().pow(2)

            loss = train_nll.clone()
            loss = loss + args.l2_regularizer_weight * weight_norm
            penalty_weight = (args.penalty_weight
                if epoch >= args.penalty_anneal_iters else 0.0)
            train_penalty = torch.max(torch.tensor(0.0).cuda(args.device), train_penalty.cuda(args.device))
            loss = loss +penalty_weight * train_penalty
            if penalty_weight > 1.0:
                loss = loss/(1. + penalty_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            pbar = tqdm(total=train_idx.size(0))
            pbar.set_description(f'Epoch {epoch:02d}')

            for batch_size, n_id, adjs in train_loader:
                # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
                adjs = [adj.to(device) for adj in adjs]

                optimizer.zero_grad()
                out = model(dataset, adjs, dataset.graph['node_feat'][n_id])
                out = F.log_softmax(out, dim=1)
                loss = criterion(out, dataset.label.squeeze(1)[n_id[:batch_size]])
                loss.backward()
                optimizer.step()
                pbar.update(batch_size)
            pbar.close()

        if args.scheduler:
            scheduler.step()
        if args.method == 'mlpnorm':
            result = evaluate_mlpnorm_hom(model, x, adj, dataset, split_idx, test_max_idx, test_min_idx, eval_func,
                            sampling=args.sampling, subgraph_loader=subgraph_loader)
        else:
            result = evaluate(model, dataset, split_idx, test_max_idx, test_min_idx,eval_func, sampling=args.sampling, subgraph_loader=subgraph_loader)
        logger.add_result(run, result[:-1])
        models_dir = f'models'


        if result[1] > best_val:
            best_val = result[1]
            early_stop_count = 0
            if args.dataset != 'ogbn-proteins':
                best_out = F.softmax(result[-1], dim=1)
            else:
                best_out = result[-1]
            # save_obj = {
            # 'model': model.state_dict(),
            # 'num_layers': args.num_layers,
            # 'hidden_channels': args.hidden_channels}
            # torch.save(save_obj, os.path.join(models_dir, f'{args.dataset}_'+f'checkpoint_best_zin.pth'))
        else:
            early_stop_count +=1
        if early_stop_count > args.early_stop:
            break
        
        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'Test_Max: {100 * result[3]:.2f}%, '
                  f'Test_Min: {100 * result[4]:.2f}%')
            if args.print_prop:
                pred = out.argmax(dim=-1, keepdim=True)
                print("Predicted proportions:", pred.unique(return_counts=True)[1].float()/pred.shape[0])

    logger.print_statistics(run)
    if args.method == 'cs':
        torch.save(best_out, f'{model_dir}/{run}.pt')
        _, out_cs = double_correlation_autoscale(dataset.label, best_out.cpu(),
            split_idx, DAD, 0.5, 50, DAD, 0.5, 50, num_hops=args.hops)
        result = evaluate(model, dataset, split_idx, eval_func, out_cs)
        cs_logger.add_result(run, (), (result[1], result[2]))

### Save results ###
if args.method == 'cs':
    print('Valid acc -> Test acc')
    res = cs_logger.display()
    best_val, best_test = res[:, 0], res[:, 1]
else:
    best_val, best_test, best_test_max, best_test_min = logger.print_statistics()
filename = f'results/HEI_{args.method}_{args.dataset}'+f'_{args.distance_method}_similarity.csv'
print(f"Saving results to {filename}")
with open(f"{filename}", 'a+') as write_obj:
    sub_dataset = f'{args.sub_dataset},' if args.sub_dataset else ''
    write_obj.write(f"{args.method}," + f"{sub_dataset}" +
                    f"{best_val.mean():.3f} ± {best_val.std():.3f}," +
                    f"{best_test.mean():.3f} ± {best_test.std():.3f}," +
                    f"{best_test_max.mean():.3f} ± {best_test_max.std():.3f}," +
                    f"{best_test_min.mean():.3f} ± {best_test_min.std():.3f}," +
                    f"{args}\n")


