import logging

import torch
from torch.autograd import Variable

from src.implem.EdgeCostEvaluator import EdgeCostEvaluator
from src.interfaces.NetworkBlock import NetworkBlock

logger = logging.getLogger(__name__)


class TimeCostEvaluator(EdgeCostEvaluator):

    def init_costs(self, model, main_cost):
        input = (Variable(torch.ones(1, *model.input_size), volatile=True),)
        graph = model.net

        self.costs = torch.Tensor(graph.number_of_nodes())

        graph.node[model.in_node]['input'] = [*input]
        for node in model.traversal_order:
            cur_node = graph.node[node]
            input = model.format_input(cur_node['input'])

            cur_mod = cur_node['module']
            out = cur_mod(input)

            if isinstance(cur_node['module'], NetworkBlock):
                cost = cur_node['module'].get_exec_time(input)
            else:
                raise RuntimeError

            logger.info('Cost for {}: {}s'.format(node, cost))
            if main_cost:
                cur_node['cost'] = cost
            self.costs[self.path_recorder.node_index[node]] = cost
            cur_node['input'] = []

            for succ in graph.successors(node):
                if 'input' not in graph.node[succ]:
                    graph.node[succ]['input'] = []
                graph.node[succ]['input'].append(out)
