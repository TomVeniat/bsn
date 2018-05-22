import networkx as nx
import numpy as np
from torch import nn

from src.interfaces.NetworkBlock import ConvBn, NetworkBlock, Add_Block, Upsamp_Block
from src.networks.SuperNetwork import SuperNetwork
from src.utils import loss
from src.utils.drawers.BSNDrawer import BSNDrawer


def downsampling_layer(n_chan, k_size, bias):
    return ConvBn(n_chan, n_chan, relu=False, k_size=k_size, stride=2, bias=bias)


def samesampling_layer(n_chan, k_size, bias):
    return ConvBn(n_chan, n_chan, relu=False, k_size=k_size, bias=bias)


def upsampling_layer(n_chan, k_size, bias):
    return Upsamp_Block(n_chan, n_chan, False, k_size, bias)


class Out_Layer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_shape, bias=True):
        super(Out_Layer, self).__init__()
        self.conv = nn.Conv2d(in_chan, out_shape[0], 1, bias=bias)
        self.out_shape = out_shape

    def forward(self, x):
        x = self.conv(x)
        return x.view(-1, *self.out_shape)

    def get_flop_cost(self, x):
        y = self.conv(x)
        return self.get_conv2d_flops(x, y, self.conv.kernel_size)


class ConvolutionalNeuralFabric(SuperNetwork):
    _NODE_FORMAT = 'A-{}_{}'
    _TRANSFORM_FORMAT = 'C-{}_{}-{}_{}'

    def __init__(self, n_layer, n_chan, data_prop, kernel_size=3, bias=True):
        super(ConvolutionalNeuralFabric, self).__init__()
        self.n_layer = n_layer
        self.n_chan = n_chan
        self.n_scale = int(np.log2(data_prop['img_dim']) + 1)
        self.in_size = data_prop['img_dim']
        self.in_chan = data_prop['in_channels']
        self.out_size = data_prop['out_size']
        self.out_dim = len(data_prop['out_size'])
        self.kernel_size = kernel_size
        self.bias = bias

        self._input_size = (self.in_chan, self.in_size, self.in_size)

        conv_params = (self.n_chan, self.kernel_size, self.bias)
        self.downsampling = lambda: downsampling_layer(*conv_params)
        self.samesampling = lambda: samesampling_layer(*conv_params)
        self.upsampling = lambda: upsampling_layer(*conv_params)

        self.is_classif = self.out_dim == 1

        if self.is_classif:
            self.loss = loss.cross_entropy_sample
        else:
            self.loss = loss.segmentation_cross_entropy

        self.blocks = nn.ModuleList([])
        self.graph = nx.DiGraph()

        # first layer
        in_name = self.init_first_layer()

        for i in range(1, self.n_layer):
            self.add_layer(i)

        if self.n_layer > 1:
            # Add vertical connections in the last layers :
            self.add_zip_layer(self.n_layer - 1, down=self.is_classif)

        out_name = self.add_output_layer()
        self.set_graph(self.graph, in_name, out_name)

    def init_first_layer(self):
        position = (0, 0)
        in_module = ConvBn(self.in_chan, self.n_chan, relu=True, bias=self.bias)
        input_node = self.add_aggregation(position, module=in_module)

        for j in range(1, self.n_scale):
            position = (0, j)
            self.add_aggregation(position, module=Add_Block())

        self.add_zip_layer(0)
        return input_node

    def add_layer(self, layer):
        in_layer = layer - 1

        for scale in range(self.n_scale):
            self.add_aggregation((layer, scale), Add_Block())

            min_scale = np.max([0, scale - 1])
            max_scale = np.min([self.n_scale, scale + 2])
            for k in range(min_scale, max_scale):
                if k < scale:  # The input has a finer scale -> downsampling
                    module = self.downsampling()
                if k == scale:  # The input has the same scale -> samesampling
                    module = self.samesampling()
                if k > scale:  # The input has a coarser scale -> upsampling
                    module = self.upsampling()

                self.add_transformation((in_layer, k), (layer, scale), module)

    def add_zip_layer(self, layer, down=True):
        module = self.downsampling if down else self.upsampling

        for j in range(0, self.n_scale - 1):
            node1 = (layer, j)
            node2 = (layer, j + 1)
            src, dst = (node1, node2) if down else (node2, node1)
            self.add_transformation(src, dst, module())

    def add_output_layer(self):
        last_layer = self.n_layer - 1
        if self.is_classif:
            out_position = (last_layer, self.n_scale - 1)
        else:
            out_position = (last_layer, 0)

        out_conv_name = self._NODE_FORMAT.format(*out_position)
        out_name = 'Lin-{}_{}-out'.format(*out_position)

        out_module = Out_Layer(self.n_chan, self.out_size, self.bias)
        self.graph.add_node(out_name, module=out_module)
        self.graph.add_edge(out_conv_name, out_name)
        self.blocks.append(out_module)

        return out_name

    def add_transformation(self, source, dest, module):
        src_l, src_s = source
        dst_l, dst_s = dest

        trans_name = self._TRANSFORM_FORMAT.format(src_l, src_s, dst_l, dst_s)
        source_name = self._NODE_FORMAT.format(src_l, src_s)
        dest_name = self._NODE_FORMAT.format(dst_l, dst_s)

        self.graph.add_node(trans_name, module=module)
        self.graph.add_edge(source_name, trans_name)
        self.graph.add_edge(trans_name, dest_name)
        self.blocks.append(module)
        return trans_name

    def add_aggregation(self, pos, module):
        agg_node_name = self._NODE_FORMAT.format(*pos)
        self.graph.add_node(agg_node_name, module=module, pos=BSNDrawer.get_draw_pos(pos=pos))
        return agg_node_name
