from torch import optim

from src.implem.AllocationCostEvaluator import AllocationCostEvaluator
from src.implem.BudgetedSuperNetwork import BudgetedSuperNetwork
from src.implem.ComputationalCostEvaluator import ComputationalCostEvaluator
from src.implem.ParameterCostEvaluator import ParameterCostEvaluator
from src.implem.ResCNF import ResCNF
from src.implem.TimeCostEvaluator import TimeCostEvaluator


def get_model(args, data_properties):
    selected_arch = args['arch']

    if selected_arch == 'CNF':
        return BudgetedSuperNetwork(static_node_proba=args['static'], n_layer=args['layers'],
                                        n_chan=args['channels'],
                                        data_prop=data_properties, deter_eval=args['deter_eval'])
    elif selected_arch == 'ResNet':
        return ResCNF(args['blocks'], args['blocks_width'], args['maps'], False, False, False,
                          args['static'], data_properties, bottlnecks=args['bottlenecks'],
                          deter_eval=args['deter_eval'])

    elif selected_arch == 'ResCNF':
        return ResCNF(args['blocks'], args['blocks_width'], args['maps'], True, True,
                          args['shift'], args['static'], data_properties, bottlnecks=args['bottlenecks'],
                          deter_eval=args['deter_eval'])

    elif selected_arch == 'FullResCNF':
        return ResCNF(args['blocks'], args['blocks_width'], args['maps'], True, True,
                          'srl', args['static'], data_properties, bottlnecks=args['bottlenecks'],
                          deter_eval=args['deter_eval'])
    else:
        raise RuntimeError


def get_cost_evaluators(args, model, path_recorder):
    cost_evaluators = {
        'comp': ComputationalCostEvaluator,
        'time': TimeCostEvaluator,
        'parallel': AllocationCostEvaluator,
        'param': ParameterCostEvaluator
    }

    model.eval()

    used_ce = {}
    for k in args['cost_evaluation']:
        if k.startswith('parallel'):
            n_para = int(k.split('_')[1])
            used_ce[k] = cost_evaluators['parallel'](path_recorder=path_recorder, n_para=n_para)
        else:
            used_ce[k] = cost_evaluators[k](path_recorder=path_recorder)
        used_ce[k].init_costs(model, main_cost=(k == args['cost_optimization']))

    return used_ce


def get_optimizer(args, model):
    if args['optim'] == 'SGD':
        optimizer = optim.SGD([
            {'params': model.blocks.parameters(), 'name': 'blocks'},
            {'params': filter(lambda x: x.requires_grad, model.sampling_parameters.parameters()),
             'name': 'path', 'lr': args['path_lr'], 'momentum': False, 'weight_decay': 0}
        ], lr=args['lr'], weight_decay=args['weight_decay'], momentum=args['momentum'], nesterov=args['nesterov'])
    elif args['optim'] == 'ADAM':
        optimizer = optim.Adam(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    elif args['optim'] == 'RMS':
        optimizer = optim.RMSprop(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'],
                                  momentum=args['momentum'])
    else:
        raise RuntimeError

    return optimizer
