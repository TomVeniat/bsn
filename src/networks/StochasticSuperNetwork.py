import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.interfaces.Observable import Observable
from src.networks.SuperNetwork import SuperNetwork


class StochasticSuperNetwork(Observable, SuperNetwork):
    INIT_NODE_PARAM = 3

    def __init__(self, deter_eval, *args, **kwargs):
        super(StochasticSuperNetwork, self).__init__(*args, **kwargs)
        self.sampled_architecture = nn.Parameter()

        self.nodes_param = None
        self.probas = None
        self.entropies = None
        self.mean_entropy = None
        self.deter_eval = deter_eval

        self.batched_sampling = None
        self.batched_log_probas = None

    def get_sampling(self, node_name, out):
        """
        Get a batch of sampling for the given node on the given output.
        Fires a "sampling" event with the node name and the sampling Variable.
        :param node_name: Name of the node to sample
        :param out: Tensor on which the sampling will be applied
        :return: A Variable brodcastable to out's size, with all dimensions equals to one except the first one (batch)
        """

        batch_size = out.size(0)
        sampling_dim = [batch_size] + [1] * (out.dim() - 1)

        static_sampling = self._get_node_static_sampling(node_name)
        if static_sampling is not None:
            sampling = out.data.new().resize_(*sampling_dim).fill_(static_sampling)
            sampling = Variable(sampling, requires_grad=False)
        else:
            node = self.net.node[node_name]

            sampling = self.batched_sampling[:, node['sampling_param']].contiguous().view(sampling_dim)

        self.fire(type='sampling', node=node_name, value=sampling)
        return sampling

    def _get_node_static_sampling(self, node_name):
        """
        Method used to check if the sampling should be done or if the node is static.
        Raise error when used with an old version of the graph.
        :param node_name:
        :return: The value of the sampling (0 or 1) if the given node is static. None otherwise.

        """
        node = self.net.node[node_name]
        if 'sampling_val' in node or 'sampling_param' not in node or node['sampling_param'] is None:
            raise RuntimeError('Deprecated method. {} node has no attribute `sampling_param` or has attribute '
                               '`sampling_val`.All nodes should now have a param '
                               '(static ones are np.inf with non trainable param).'.format(node_name))

        if not self.training and self.deter_eval:
            # Model is in deterministic evaluation mode
            if not isinstance(node['sampling_param'], int):
                return (F.sigmoid(self.sampling_parameters[node['sampling_param']]) > 0.5).item()
        return None

    def forward(self, *input):
        self.fire(type='new_iteration')

        assert len(input) == 1
        self._sample_archs(input[0].size(0))

        self.net.node[self.in_node]['input'] = [*input]

        for node in self.traversal_order:
            cur_node = self.net.node[node]
            input = self.format_input(cur_node.pop('input'))

            if len(input) == 0:
                raise RuntimeError('Node {} has no inputs'.format(node))

            out = cur_node['module'](input)

            sampling = self.get_sampling(node, out)
            out = out * sampling

            if node == self.out_node:
                return out

            for succ in self.net.successors(node):
                if 'input' not in self.net.node[succ]:
                    self.net.node[succ]['input'] = []
                self.net.node[succ]['input'].append(out)

    def _sample_archs(self, batch_size):
        params = torch.stack([p for p in self.sampling_parameters], dim=1)
        probas_resized = params.sigmoid().expand(batch_size, len(self.sampling_parameters))
        distrib = torch.distributions.Bernoulli(probas_resized)
        self.batched_sampling = distrib.sample()

        self.batched_log_probas = distrib.log_prob(self.batched_sampling)

    @property
    def n_layers(self):
        return sum([mod.n_layers for mod in self.blocks])

    @property
    def n_comp_steps(self):
        return sum([mod.n_comp_steps for mod in self.blocks])

    def reinit_sampling_params(self):
        new_params = nn.ParameterList()
        for p in self.sampling_parameters:
            if p.requires_grad:
                param_value = self.INIT_NODE_PARAM
            else:
                param_value = p.data[0]
            new_params.append(nn.Parameter(p.data.new(([param_value])), requires_grad=p.requires_grad))

        self.sampling_parameters = new_params

    def update_probas_and_entropies(self):
        if self.nodes_param is None:
            self._init_nodes_param()
        self.probas = {}
        self.entropies = {}
        self.mean_entropy = .0
        for node, props in self.graph.node.items():
            param = self.sampling_parameters[props['sampling_param']]
            p = param.sigmoid().item()
            self.probas[node] = p
            if p in [0, 1]:
                e = 0
            else:
                e = -(p * np.log2(p)) - ((1 - p) * np.log2(1 - p))
            self.entropies[node] = e
            self.mean_entropy += e
        self.mean_entropy /= self.graph.number_of_nodes()

    def _init_nodes_param(self):
        self.nodes_param = {}
        for node, props in self.graph.node.items():
            if 'sampling_param' in props and props['sampling_param'] is not None:
                self.nodes_param[node] = props['sampling_param']

    def __str__(self):
        model_descr = 'Model:{}\n\t{} nodes\n\t{} blocks\n\t{} parametrized layers\n\t{} computation steps\n\t{} parameters\n\t{} meta-params'
        return model_descr.format(type(self).__name__, self.graph.number_of_nodes(), len(self.blocks), self.n_layers,
                                  self.n_comp_steps,
                                  sum(i.numel() for i in self.parameters()), len(self.sampling_parameters))
