from models import LINK, LINK_multi,GCN, MLP, SGC, GAT, SGCMem, MultiLP, MixHop, GCNJK, GATJK, H2GCN, APPNP_Net, LINK_Concat, LINKX, LINKX_Pro, GPRGNN, GCNII,MLPNORM
from data_utils import normalize
import math
def parse_method(args, dataset, n, c, d, device):
    if args.method == 'link':
        model = LINK(n, c).to(device)
    elif args.method == 'link_multi':
        model = LINK_multi(n, args.hidden_channels, c,  args.num_layers, dropout=args.dropout).to(device)
    elif args.method == 'gcn':
        if args.dataset == 'ogbn-proteins':
            # Pre-compute GCN normalization.
            dataset.graph['edge_index'] = normalize(dataset.graph['edge_index'])
            model = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        dropout=args.dropout,
                        save_mem=True,
                        use_bn=not args.no_bn).to(device)
        else:
            model = GCN(in_channels=d,
                        hidden_channels=args.hidden_channels,
                        out_channels=c,
                        num_layers=args.num_layers,
                        dropout=args.dropout,
                        use_bn=not args.no_bn).to(device)

    elif args.method == 'mlp' or args.method == 'cs':
        model = MLP(in_channels=d, hidden_channels=args.hidden_channels,
                    out_channels=c, num_layers=args.num_layers,
                    dropout=args.dropout).to(device)
    elif args.method == 'sgc':
        if args.cached:
            model = SGC(in_channels=d, out_channels=c, hops=args.hops).to(device)
        else:
            model = SGCMem(in_channels=d, out_channels=c,
                           hops=args.hops).to(device)
    elif args.method == 'gprgnn':
        model = GPRGNN(d, args.hidden_channels, c, alpha=args.gpr_alpha, num_layers=args.num_layers, dropout=args.dropout).to(device)
    elif args.method == 'appnp':
        model = APPNP_Net(d, args.hidden_channels, c, alpha=args.gpr_alpha, dropout=args.dropout, num_layers=args.num_layers).to(device)
    elif args.method == 'gat':
        model = GAT(d, args.hidden_channels, c, num_layers=args.num_layers,
                    dropout=args.dropout, heads=args.gat_heads).to(device)
    elif args.method == 'lp':
        mult_bin = args.dataset=='ogbn-proteins'
        model = MultiLP(c, args.lp_alpha, args.hops, mult_bin=mult_bin)
    elif args.method == 'mixhop':
        model = MixHop(d, args.hidden_channels, c, num_layers=args.num_layers,
                       dropout=args.dropout, hops=args.hops).to(device)
    elif args.method == 'gcnjk':
        model = GCNJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, jk_type=args.jk_type).to(device)
    elif args.method == 'gatjk':
        model = GATJK(d, args.hidden_channels, c, num_layers=args.num_layers,
                        dropout=args.dropout, heads=args.gat_heads,
                        jk_type=args.jk_type).to(device)
    elif args.method == 'h2gcn':
        model = H2GCN(d, args.hidden_channels, c, dataset.graph['edge_index'],
                        dataset.graph['num_nodes'],
                        num_layers=args.num_layers, dropout=args.dropout,
                        num_mlp_layers=args.num_mlp_layers).to(device)
    elif args.method == 'link_concat':
        model = LINK_Concat(d, args.hidden_channels, c, args.num_layers, dataset.graph['num_nodes'], dropout=args.dropout).to(device)
    elif args.method == 'linkx':
        #######
        model = LINKX(d, args.hidden_channels, c, args.num_layers, dataset.graph['num_nodes'],
        inner_activation=args.inner_activation, inner_dropout=args.inner_dropout, dropout=args.dropout, init_layers_A=args.link_init_layers_A, init_layers_X=args.link_init_layers_X).to(device)
        ##### dssl encoder
        # model = LINKX(d, args.hidden_channels, args.hidden_channels, args.num_layers, dataset.graph['num_nodes'],
        # inner_activation=args.inner_activation, inner_dropout=args.inner_dropout, dropout=args.dropout, init_layers_A=args.link_init_layers_A, init_layers_X=args.link_init_layers_X).to(device)
    elif args.method == 'linkx_pro':
        model = LINKX_Pro(d, args.hidden_channels, c, args.num_layers, dataset.graph['num_nodes'], args.num_prototypes_per_class, enable_prot=args.enable_prot,
        inner_activation=args.inner_activation, inner_dropout=args.inner_dropout, dropout=args.dropout, init_layers_A=args.link_init_layers_A, init_layers_X=args.link_init_layers_X).to(device)
    elif args.method == 'gcn2':
        model = GCNII(d, args.hidden_channels, c, args.num_layers, args.gcn2_alpha, args.theta, dropout=args.dropout).to(device)
    elif args.method == 'mlpnorm':
        model = MLPNORM(nnodes=dataset.graph['num_nodes'], nfeat=d, nhid=args.hidden_channels, nclass=c, dropout=args.dropout, alpha=args.alpha, beta=args.beta, gamma=args.gamma,
                        delta=args.delta, norm_func_id=args.norm_func_id, norm_layers=args.norm_layers, orders_func_id=args.orders_func_id, orders=args.orders, device=device).to(device)
    else:
        raise ValueError('Invalid method')
    return model


def parser_add_main_args(parser):
    parser.add_argument('--dataset', type=str, default='fb100')
    parser.add_argument('--sub_dataset', type=str, default='')
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--method', '-m', type=str, default='link')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--hops', type=int, default=1,
                        help='power of adjacency matrix for certain methods')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers for deep methods')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--cached', action='store_true',
                        help='set to use faster sgc')
    parser.add_argument('--gat_heads', type=int, default=8,
                        help='attention heads for gat')
    parser.add_argument('--lp_alpha', type=float, default=.1,
                        help='alpha for label prop')
    parser.add_argument('--gpr_alpha', type=float, default=.1,
                        help='alpha for gprgnn')
    parser.add_argument('--gcn2_alpha', type=float, default=.1,
                        help='alpha for gcn2')
    parser.add_argument('--theta', type=float, default=.5,
                        help='theta for gcn2')
    parser.add_argument('--directed', action='store_true',
                        help='set to not symmetrize adjacency')
    parser.add_argument('--jk_type', type=str, default='max', choices=['max', 'lstm', 'cat'],
                        help='jumping knowledge type')
    parser.add_argument('--rocauc', action='store_true',
                        help='set the eval function to rocauc')
    parser.add_argument('--num_mlp_layers', type=int, default=1,
                        help='number of mlp layers in h2gcn')
    parser.add_argument('--print_prop', action='store_true',
                        help='print proportions of predicted class')
    parser.add_argument('--train_prop', type=float, default=.5,
                        help='training label proportion')
    parser.add_argument('--valid_prop', type=float, default=.25,
                        help='validation label proportion')
    parser.add_argument('--adam', action='store_true', help='use adam instead of adamW')
    parser.add_argument('--rand_split', action='store_true', help='use random splits')
    parser.add_argument('--no_bn', action='store_true', help='do not use batchnorm')
    parser.add_argument('--sampling', action='store_true', help='use neighbor sampling')
    parser.add_argument('--inner_activation', action='store_true', help='Whether linkV3 uses inner activation')
    parser.add_argument('--inner_dropout', action='store_true', help='Whether linkV3 uses inner dropout')
    parser.add_argument("--SGD", action='store_true', help='Use SGD as optimizer')
    parser.add_argument('--link_init_layers_A', type=int, default=1)
    parser.add_argument('--link_init_layers_X', type=int, default=1)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--temp", type=str, default='')
    parser.add_argument("--seeds", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.5)
    parser.add_argument("--value", type=int, default= 5)
    parser.add_argument("--rn_base_weight", type=float, default=0.5)
    parser.add_argument("--rn_scale_weight", type=float, default=0.5)
    parser.add_argument("--distance_weight", type=float, default=0.5)
    parser.add_argument("--cl_weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2)
    parser.add_argument("--tmp", type=int, default= 0)
    parser.add_argument("--weight", type=int, default= 1)
    parser.add_argument("--num_prototypes_per_class", type=int, default= 10)
    parser.add_argument("--enable_prot", type=str, default= True)
    parser.add_argument("--prototype_cal", type=str, default= "mean")####### mean, distance
    parser.add_argument("--prototype_type", type=str, default= "feat")###### feat,logit
    parser.add_argument("--prototype_loss", type=str, default= "positive")###### positive,global
    parser.add_argument("--early_stop", type=int, default= 100)
    parser.add_argument("--warm_up_epochs", type=int, default= 10)

       # used for mlpnorm
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='Weight for frobenius norm on Z.')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for frobenius norm on Z-A')
    parser.add_argument('--gamma', type=float, default=0.0,
                        help='Weight for MLP results kept')
    parser.add_argument('--delta', type=float, default=0.0,
                        help='Weight for node features, thus 1-delta for adj')
    parser.add_argument('--norm_func_id', type=int, default=2,
                        help='Function of norm layer, ids \in [1, 2]')
    parser.add_argument('--norm_layers', type=int, default=1,
                        help='Number of groupnorm layers')
    parser.add_argument('--orders_func_id', type=int, default=2,
                        help='Sum function of adj orders in norm layer, ids \in [1, 2, 3]')
    parser.add_argument('--orders', type=int, default=1,
                        help='Number of adj orders in norm layer')
    # # used for ggcn
    # parser.add_argument('--decay_rate', type=float, default=1.0,
    #                     help='decay_rate in the decay function')
    # parser.add_argument('--exponent', type=float, default=3.0,
    #                     help='exponent in the decay function')

    # # used for wrgat
    # parser.add_argument("--original_edges", default=False, action='store_true')
    # parser.add_argument("--original_edges_weight", type=float, default=1.0)
    # parser.add_argument("--filter_structure_relation",
    #                     default=False, action='store_true')
    # parser.add_argument(
    #     "--filter_structure_relation_number", type=int, default=10)
    # parser.add_argument("--st_thres", type=float,  default=-
    #                     math.inf, help="edge weight threshold")
