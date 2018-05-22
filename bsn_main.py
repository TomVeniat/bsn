import argparse
import logging
import os

import logger as data_logger
import torch
from torch.autograd import Variable
from tqdm import tqdm

import src.utils.external_resources as external
from src.interfaces.PathRecorder import PathRecorder
from src.utils import initializers
from src.utils.datasets import get_data
from src.utils.drawers import drawing_utils
from src.utils.misc import restricted_float, restricted_str, restricted_list
from src.utils.optimization import adjust_lr

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def evaluate_model(model, x, y, dataset_loader, name='Test'):
    correct = 0
    total = 0
    for images, labels in tqdm(dataset_loader, desc=name, ascii=True):
        x.resize_(images.size()).copy_(images)
        y.resize_(labels.size()).copy_(labels)

        with torch.no_grad():
            preds = model(x)

        _, predicted = torch.max(preds.data, 1)
        total += labels.size(0)

        correct += (predicted == y).sum()

    return 100 * correct / total


def argument_parser():
    parser = argparse.ArgumentParser(description='Budgeted Super Networks')

    # Experience
    parser.add_argument('-exp-name', action='store', default='', type=str, help='Experience Name')
    # Model
    parser.add_argument('-arch', action='store', default='ResNet',
                        type=restricted_str('CNF', 'ResNet', 'MSDNet', 'ResCNF', 'FullResCNF', ))
    parser.add_argument('-deter_eval', action='store', default=True, type=bool,
                        help='Take blocks with probas >0.5 instead of sampling during evaluation')
    # BSN and MSDNet Args
    parser.add_argument('-layers', dest='layers', action='store', default=3, type=int,
                        help='Number of layers')
    parser.add_argument('-channels', dest='channels', action='store', default=5, type=int,
                        help='Number of channels')

    # ResCNF Args
    parser.add_argument('-blk', dest='blocks', action='store', default=3, type=int, help='Number of block')
    parser.add_argument('-blk_w', dest='blocks_width', nargs='*', action='store', default=[3], type=int,
                        help='Number of element per block')
    parser.add_argument('-shift', dest='shift', action='store', default='slr', type=str,
                        help='Connect layers with a shift in the shortcut connections')
    parser.add_argument('-shortcuts', dest='shortcuts', action='store', default=True, type=bool,
                        help='True to add shortcuts to the network false for classic resnet')
    parser.add_argument('-shortcuts_res', dest='shortcuts_res', action='store', default=True, type=bool,
                        help='Add a skip connection to the shortcut connection')
    parser.add_argument('-maps', action='store', nargs='*', default=[16, 32, 64], type=str, help='maps per block')
    parser.add_argument('-bottlenecks', action='store', default=False, type=bool, help='Use bottleneck nodes')

    # Training
    parser.add_argument('-path', default='./data/', type=str,
                        help='path for the execution')

    parser.add_argument('-dset', default='CIFAR10', type=str, help='Dataset')
    parser.add_argument('-bs', action='store', default=64, type=int, help='Size of each batch')
    parser.add_argument('-epochs', action='store', default=300, type=int,
                        help='Number of training epochs')

    parser.add_argument('-optim', action='store', default='SGD', type=str,
                        help='Optimization method')
    parser.add_argument('-nesterov', action='store', default=False, type=bool,
                        help='Use Nesterov for SGD momentum')
    parser.add_argument('-lr', action='store', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('-path_lr', action='store', default=1e-3, type=float, help='path learning rate')
    parser.add_argument('-momentum', action='store', default=0.9, type=float,
                        help='momentum used by the optimizer')
    parser.add_argument('-wd', dest='weight_decay', action='store', default=1e-4, type=float,
                        help='weight decay used during optimisation')

    parser.add_argument('-lr_pol_tresh', action='store', default=[150, 225], type=str,
                        help='learning rate decay rate')
    parser.add_argument('-lr_pol_val', action='store', nargs='*', default=[0.1, 0.01, 0.001], type=str,
                        help='learning rate decay period')

    parser.add_argument('-cuda', action='store', default=-1, type=int,
                        help='Enables cuda and select device')

    parser.add_argument('-lp', action='store', default=-1, type=int,
                        help='Number of iterations between two logging messages')
    parser.add_argument('-draw_env', default=None, type=str, help='Visdom drawing environment')

    parser.add_argument('-static', action='store', default=-1, type=restricted_float(0, 1),
                        help='sample a static binary weight with given proba for each stochastic Node.')

    parser.add_argument('-np', '--n_parallel', dest='n_parallel', action='store', default=3, type=int,
                        help='Maximum number of module evaluation in parallel')
    parser.add_argument('-ce', '-cost_evaluation', dest='cost_evaluation', action='store',
                        default=['comp', 'parallel_2'],
                        type=restricted_list('comp', 'time', 'para'))
    parser.add_argument('-co', dest='cost_optimization', action='store', default='comp',
                        type=restricted_str('comp', 'time', 'para'))

    parser.add_argument('-lambda', dest='lambda', action='store', default=1e-7, type=float,
                        help='Constant balancing the ratio classifier loss/architectural loss')
    parser.add_argument('-oc', dest='objective_cost', action='store', default=0, type=float,
                        help='Maximum allowed cost for architecture')
    parser.add_argument('-om', dest='objective_method', action='store', default='max',
                        type=restricted_str('max', 'abs'), help='Method used to compute the cost of an architecture')
    parser.add_argument('-pen', dest='arch_penalty', action='store', default=0, type=float,
                        help='Penalty for inconsistent architecture')

    return parser.parse_known_args()[0]


def init_expe(args, data_properties):
    model = initializers.get_model(args, data_properties)
    optimizer = initializers.get_optimizer(args, model)

    path_recorder = PathRecorder(model.graph, model.out_node)
    model.subscribe(path_recorder.new_event)

    cost_evaluators = initializers.get_cost_evaluators(args, model, path_recorder)

    return model, optimizer, path_recorder, cost_evaluators


def main(args, data_logger):
    logger.info('Starting run : {}'.format(args['exp_name']))
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args['cuda'])

    train_loader, val_loader, test_loader, data_properties = get_data(args['dset'], args['bs'], args['path'], args)

    if args['lp'] == -1:
        args['lp'] = len(train_loader)

    model, optimizer, path_recorder, cost_evaluators = init_expe(args, data_properties)

    logger.info(model)
    drawer = drawing_utils.get_drawer(model, args['draw_env'])
    draw_ops = dict(width=600, height=450)

    train_logger = data_logger.ParentWrapper(tag='train', name='parent',
                                             children=(data_logger.AvgMetric(name='classif_loss'),
                                                       data_logger.AvgMetric(name='accuracy'),
                                                       data_logger.AvgMetric(name='rewards'),
                                                       data_logger.AvgMetric(name='cost_loss')))
    data_logger.SimpleMetric(name='lambda')
    data_logger.SimpleMetric(name='entropy')
    data_logger.SimpleMetric(name='accuracy', tag='val')
    data_logger.SimpleMetric(name='accuracy', tag='test')
    data_logger.SimpleMetric(name='objective_cost', tag='target')
    data_logger.AvgMetric(name='objective_cost', tag='real')

    for cost in cost_evaluators:
        data_logger.AvgMetric(name=cost, tag='train_sampled')
        data_logger.AvgMetric(name=cost, tag='train_pruned')
        data_logger.SimpleMetric(name=cost, tag='eval_sampled')
        data_logger.SimpleMetric(name=cost, tag='eval_pruned')

    tot_costs = dict(('total_cost_' + ce_name, ce.total_cost) for ce_name, ce in cost_evaluators.items())
    logger.info(tot_costs)

    x = torch.Tensor()
    y = torch.LongTensor()

    if args['cuda'] > -1:
        logger.info('Running with cuda (GPU n°{})'.format(args['cuda']))
        model.cuda()
        x = x.cuda()
        y = y.cuda()
    else:
        logger.warning('Running *WITHOUT* cuda')

    drawer.draw(model.graph, param_list=model.sampling_parameters, vis_opts={'title': 'Probas', **draw_ops},
                vis_win='Probas')
    drawer.draw_weights(model.graph, vis_opts={'title': 'Costs', **draw_ops}, vis_win='Costs')

    for epoch in range(args['epochs']):
        logger.info(epoch)
        adjust_lr(optimizer, epoch, args['lr_pol_tresh'], args['lr_pol_val'], logger, ['path'])

        for ce in cost_evaluators.values():
            ce.new_epoch()

        for i, (inputs, labels) in enumerate(tqdm(train_loader, desc='Train', ascii=True)):
            model.train()
            nelem = inputs.size(0)

            x.resize_(inputs.size()).copy_(inputs)
            y.resize_(labels.size()).copy_(labels)

            predictions = model(Variable(x))

            _, predicted = torch.max(predictions.data, 1)
            correct = (predicted == y).sum().item()

            indiv_loss = model.loss(predictions, Variable(y))

            archs = path_recorder.get_architectures(model.out_node)

            optim_cost = None
            for cost, cost_eval in cost_evaluators.items():
                sampled_cost, pruned_cost = cost_eval.get_costs(archs)

                if cost == args['cost_optimization']:
                    optim_cost = sampled_cost
                    data_logger.get_metric(name='objective_cost', tag='real').update(sampled_cost.mean().item(),
                                                                                     n=nelem)

                data_logger.get_metric(tag='train_sampled', name=cost).update(sampled_cost.mean().item(), n=nelem)
                data_logger.get_metric(tag='train_pruned', name=cost).update(pruned_cost.mean().item(), n=nelem)

            penalty = path_recorder.get_consistence(model.out_node).float()
            cost = (optim_cost + args['arch_penalty'] * penalty) - args['objective_cost']

            if args['objective_method'] == 'max':
                cost.clamp_(min=0)
            elif args['objective_method'] == 'abs':
                cost.abs_()
            else:
                raise RuntimeError

            data_logger.get_metric(name='cost_loss', tag='train').update(cost.mean().item(), n=nelem)
            cost = indiv_loss.data.new(cost.size()).copy_(cost)

            rewards = -(indiv_loss.data.squeeze() + args['lambda'] * cost)
            mean_reward = rewards.mean()
            rewards = (rewards - mean_reward)
            data_logger.get_metric(tag='train', name='rewards').update(mean_reward.item(), n=nelem)
            rewards = Variable(rewards)

            sampling_loss = (-model.batched_log_probas * rewards.unsqueeze(dim=1)).sum()
            loss = indiv_loss.mean() + sampling_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_logger.update(classif_loss=loss.item(),
                                accuracy=correct * 100 / nelem,
                                rewards=mean_reward.item(),
                                n=nelem)

            if (i + 1) % args['lp'] == 0:
                logger.info('Evaluation')

                model.eval()
                model.update_probas_and_entropies()

                progress = epoch + (i + 1) / len(train_loader)

                data_logger.get_metric(name='lambda').update(args['lambda'])
                data_logger.get_metric(name='objective_cost', tag='target').update(args['objective_cost'])
                data_logger.get_metric(name='entropy').update(model.mean_entropy)

                val_score = evaluate_model(model, x, y, val_loader, 'validation')
                test_score = evaluate_model(model, x, y, test_loader, 'test')

                data_logger.Accuracy_Val.update(val_score.item())
                data_logger.Accuracy_Test.update(test_score.item())

                archs = path_recorder.get_architectures(model.out_node)
                for ce_name, ce in cost_evaluators.items():
                    samp_cost, pruned_cost = ce.get_costs(archs)
                    data_logger.get_metric(tag='eval_sampled', name=ce_name).update(samp_cost.mean().item(), n=1)
                    data_logger.get_metric(tag='eval_pruned', name=ce_name).update(pruned_cost.mean().item(), n=1)

                drawer.scatter(model.probas, model.entropies, opts={'title': 'Proba distibutions'},
                               vis_win='distrib')

                real_paths, sampling_paths = path_recorder.get_graph_paths(model.out_node)
                drawer.draw(model.graph, param_list=model.sampling_parameters, vis_opts={'title': 'Probas', **draw_ops},
                            vis_win='probas')
                drawer.draw(model.graph, vis_opts={'title': 'Full', **draw_ops}, weights=sampling_paths[0],
                            vis_win='full')

                data_logger.log_with_tag(tag='*', idx=progress, reset=True)
                msg = '[{:.2f}] Loss: {:.5f} - Cost: {:.3E} - Val: {:2.2f}% - Test: {:2.2f}% - Train: {:2.2f}%'
                logger.info(msg.format(progress,
                                       data_logger.classif_loss_train,
                                       data_logger.objective_cost_real,
                                       data_logger.accuracy_val,
                                       data_logger.accuracy_test,
                                       data_logger.accuracy_train))

                if len(real_paths[0]) > 0:
                    sg = model.graph.subgraph(real_paths[0])

                    drawer.draw(sg, vis_opts={'title': 'Real', **draw_ops}, weights=1.0, vis_win='Clean')

                    for ce_name, ce in cost_evaluators.items():
                        if ce_name.startswith('parallel'):
                            n_steps = max(ce.node_steps[0].values())
                            drawer.draw(sg, vis_opts={'title': 'Alloc ' + ce_name, **draw_ops},
                                        weights=ce.node_steps[0],
                                        colormap=drawing_utils.get_colormap(n_steps),
                                        vis_win='Alloc ' + ce_name)

                drawer.draw(model.graph, vis_opts={'title': 'Mean', **draw_ops},
                            weights=path_recorder.get_posterior_weights(), vis_win='mean')

        logger.info('EPOCH DONE')


if __name__ == '__main__':
    logger.info('Executing main from {}'.format(os.getcwd()))
    exp_name = 'Direct Launch'
    args = vars(argument_parser())

    vis_conf = external.get_visdom_conf()
    xp_logger = data_logger.Experiment(exp_name, use_visdom=True,
                                       visdom_opts={'server': vis_conf['url'], 'port': vis_conf['port'],
                                                    'env': args['draw_env']},
                                       time_indexing=False, xlabel='Epoch')
    main(args, xp_logger)
