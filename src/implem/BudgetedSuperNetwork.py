import matplotlib.pyplot as plt
import numpy as np
from torch import nn, torch

from src.implem.ConvolutionalNeuralFabrics import ConvolutionalNeuralFabric, Out_Layer
from src.networks.StochasticSuperNetwork import StochasticSuperNetwork
from src.utils.drawers.BSNDrawer import BSNDrawer

plt.switch_backend('agg')


class BudgetedSuperNetwork(StochasticSuperNetwork, ConvolutionalNeuralFabric):
    def __init__(self, static_node_proba, *args, **kwargs):
        self.static_node_proba = static_node_proba
        self.sampling_parameters = []
        super(BudgetedSuperNetwork, self).__init__(*args, **kwargs)
        self.sampling_parameters = nn.ParameterList(self.sampling_parameters)
        # self.set_graph(self.graph, 'In', 'Out')

    def add_transformation(self, source, dest, module):
        src_l, src_s = source
        dst_l, dst_s = dest

        trans_name = self._TRANSFORM_FORMAT.format(src_l, src_s, dst_l, dst_s)
        source_name = self._NODE_FORMAT.format(src_l, src_s)
        dest_name = self._NODE_FORMAT.format(dst_l, dst_s)

        pos = BSNDrawer.get_draw_pos(source=source, dest=dest)

        sampling_param = self.sampling_param_generator(trans_name)

        self.graph.add_node(trans_name, module=module, sampling_param=len(self.sampling_parameters), pos=pos)
        self.graph.add_edge(source_name, trans_name, width_node=trans_name)
        self.graph.add_edge(trans_name, dest_name, width_node=trans_name)

        self.sampling_parameters.append(sampling_param)
        self.blocks.append(module)
        return trans_name

    def add_aggregation(self, pos, module):
        agg_node_name = self._NODE_FORMAT.format(*pos)
        sampling_param = self.sampling_param_generator(agg_node_name)

        self.graph.add_node(agg_node_name, module=module, sampling_param=len(self.sampling_parameters),
                            pos=BSNDrawer.get_draw_pos(pos=pos))


        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)
        self.blocks.append(module)
        return agg_node_name

    def add_output_layer(self):
        last_layer = self.n_layer - 1
        out_scale = (self.n_scale - 1) if self.is_classif else 0

        out_features_name = self._NODE_FORMAT.format(last_layer, out_scale)

        out_pos = (last_layer + 1, out_scale)
        out_name = 'Lin-{}_{}-out'.format(*out_pos)

        out_module = Out_Layer(self.n_chan, self.out_size, self.bias)

        sampling_param = self.sampling_param_generator(out_name)

        self.graph.add_node(out_name, module=out_module, sampling_param=len(self.sampling_parameters),
                            pos=BSNDrawer.get_draw_pos(pos=out_pos))
        self.graph.add_edge(out_features_name, out_name, width_node=out_name)

        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)
        self.blocks.append(out_module)

        return out_name

    def sampling_param_generator(self, node_name):
        if not node_name.startswith('C'):
            param_value = np.inf
            trainable = False
        elif self.static_node_proba >= 0:
            param_value = 1 if np.random.rand() < self.static_node_proba else -1
            param_value *= np.inf
            trainable = False
        else:
            # Node is a convolution
            param_value = 3
            trainable = True

        return nn.Parameter(torch.Tensor([param_value]), requires_grad=trainable)
