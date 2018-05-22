from numbers import Number

import networkx as nx
import numpy as np
import torch.nn.functional as F
from torch import nn, torch

from src.interfaces.NetworkBlock import ConvBn, NetworkBlock, Add_Block
from src.networks.StochasticSuperNetwork import StochasticSuperNetwork
from src.utils import loss
from src.utils.drawers.ResCNFDrawer import ResCNFDrawer


class BasicBlock(NetworkBlock):
    n_layers = 2
    n_comp_steps = 1

    def __init__(self, in_chan, out_chan, bias, **kwargs):
        super(BasicBlock, self).__init__()
        assert in_chan <= out_chan
        stride = int(out_chan / in_chan)
        self.conv1 = ConvBn(in_chan, out_chan, stride=stride, relu=True, bias=bias)
        self.conv2 = ConvBn(out_chan, out_chan, relu=False, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(x)

    def get_flop_cost(self, x1):
        x2 = self.conv1(x1)
        y = self.conv2(x2)

        cost = self.get_conv2d_flops(x1, x2, self.conv1.conv.kernel_size)
        cost += self.get_conv2d_flops(x2, y, self.conv2.conv.kernel_size)

        assert cost == self.conv1.get_flop_cost(x1) + self.conv2.get_flop_cost(x2)

        return cost


class BottleneckBlock(NetworkBlock):
    n_layers = 3
    n_comp_steps = 1

    def __init__(self, input_chan, out_chan, bias, stride=None, use_stride=True, bottleneck_factor=4):
        super(BottleneckBlock, self).__init__()
        assert input_chan <= out_chan
        stride = stride if stride is not None else int(out_chan / input_chan) if use_stride else 1
        inside_chan = int(out_chan / bottleneck_factor)
        self.conv1 = ConvBn(input_chan, inside_chan, k_size=1, stride=stride, padding=0, relu=True, bias=bias)
        self.conv2 = ConvBn(inside_chan, inside_chan, relu=True, bias=bias)
        self.conv3 = ConvBn(inside_chan, out_chan, k_size=1, padding=0, relu=False, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        y = self.conv3(x)
        return y

    def get_flop_cost(self, x1):
        x2 = self.conv1(x1)
        x3 = self.conv2(x2)
        y = self.conv3(x3)

        cost = self.get_conv2d_flops(x1, x2, self.conv1.conv.kernel_size)
        cost += self.get_conv2d_flops(x2, x3, self.conv2.conv.kernel_size)
        cost += self.get_conv2d_flops(x3, y, self.conv3.conv.kernel_size)

        assert cost == self.conv1.get_flop_cost(x1) + self.conv2.get_flop_cost(x2) + self.conv3.get_flop_cost(x3)

        return cost


class Skip_Block(NetworkBlock):
    def __init__(self, in_chan, out_chan, bias, stride=None, use_stride=True):
        super(Skip_Block, self).__init__()
        assert in_chan <= out_chan
        self.projection = None
        if in_chan != out_chan:
            stride = stride if stride is not None else int(out_chan / in_chan) if use_stride else 1
            self.projection = ConvBn(in_chan, out_chan, relu=False, k_size=1, padding=0, stride=stride, bias=bias)
            # self.projection = ConvBn(in_chan, out_chan, relu=False, stride=stride, bias=bias)

    def forward(self, x):
        if self.projection is not None:
            x = self.projection(x)
        return x

    def get_flop_cost(self, x):
        if self.projection is None:
            return 0
        else:
            return self.projection.get_flop_cost(x)

    @property
    def n_layers(self):
        return 0 if self.projection is None else 1

    @property
    def n_comp_steps(self):
        return 0 if self.projection is None else 1


class Out_Layer(NetworkBlock):
    n_layers = 1
    n_comp_steps = 1

    def __init__(self, in_chan, out_dim, bias=True):
        super(Out_Layer, self).__init__()
        self.fc = nn.Linear(in_chan, out_dim, bias=bias)

    def forward(self, x):
        assert x.size(-1) == 8
        x = F.avg_pool2d(x, x.size(-1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def get_flop_cost(self, x):
        return self.fc.in_features * self.fc.out_features


class ResCNF(StochasticSuperNetwork):
    INPUT_NAME = 'Input'
    OUTPUT_NAME = 'Output'
    CLASSIC_BLOCK_NAME = 'CLASSIC_B{}_N{}'
    SHORTCUT_BLOCK_NAME = 'SHORTCUT_B{}_N{}_{}'
    CLASSIC_SKIP_NAME = 'SKIP_CL_B{}_N{}'
    SHORTCUT_SKIP_NAME = 'SKIP_SH_B{}_N{}_{}'
    ADD_NAME = 'ADD_B{}_N{}'
    IDENT_NAME = 'IDENT_B{}'

    def __init__(self, layers, blocks_per_layer, n_channels, shortcuts, shortcuts_res, shift, static_node_proba,
                 data_prop, bottlnecks, bn_factor=4, bias=True, *args, **kwargs):
        super(ResCNF, self).__init__(*args, **kwargs)
        assert len(n_channels) == layers and (len(blocks_per_layer) == 1 or len(blocks_per_layer) == layers)
        self.in_chan = data_prop['in_channels']
        self.in_size = data_prop['img_dim']
        self.out_dim = data_prop['out_size'][0]
        self.loss = loss.cross_entropy_sample
        self.static_node_proba = static_node_proba

        self._input_size = (self.in_chan, self.in_size, self.in_size)

        self.blocks = nn.ModuleList([])
        self.graph = nx.DiGraph()
        self.sampling_parameters = nn.ParameterList()

        self.bottleneck = bottlnecks
        if bottlnecks:
            self.block_type = BottleneckBlock
            self.scale_factor = bn_factor
        else:
            self.block_type = BasicBlock
            self.scale_factor = 1

        if len(blocks_per_layer) == 1:
            blocks_per_layer = blocks_per_layer * len(n_channels)

        in_node = ConvBn(self.in_chan, n_channels[0], relu=True, bias=bias)
        self.add_node([], self.INPUT_NAME, in_node, (0, 0))

        last_node = self.add_layer(0, blocks_per_layer[0], n_channels[0], n_channels[0] * self.scale_factor,
                                   self.INPUT_NAME, False,
                                   False, shift, bias, stride=False)
        for l in range(1, layers):
            in_chan = n_channels[l - 1] * self.scale_factor
            last_node = self.add_layer(l, blocks_per_layer[l], in_chan, n_channels[l] * self.scale_factor, last_node,
                                       shortcuts, shortcuts_res,
                                       shift, bias, stride=True)

        out_node = Out_Layer(n_channels[-1] * self.scale_factor, self.out_dim, bias=bias)
        self.add_node([last_node], self.OUTPUT_NAME, out_node, (layers - 1, blocks_per_layer[-1] + 1))
        self.set_graph(self.graph, self.INPUT_NAME, self.OUTPUT_NAME)

    def add_layer(self, b, n_blocks, in_chan, out_chan, last_node, shortcuts, sh_res, shift, bias, stride):
        prev_in_chan = in_chan
        for n in range(n_blocks):
            use_stride = stride and n == 0
            if b > 0 and n == 0:
                # Add identity block (only for drawing)
                ident_block = Skip_Block(in_chan, in_chan, bias)
                ident_name = self.IDENT_NAME.format(b)
                self.add_node([last_node], ident_name, ident_block, (b,))
                last_node = ident_name
            # Add basic block
            basic_block = self.block_type(in_chan, out_chan, bias=bias, use_stride=use_stride)
            basic_block_name = self.CLASSIC_BLOCK_NAME.format(b, n)
            self.add_node([last_node], basic_block_name, basic_block, (b, n))
            # Add skip connection
            skip = Skip_Block(in_chan, out_chan, bias, use_stride=use_stride)
            skip_name = self.CLASSIC_SKIP_NAME.format(b, n)
            self.add_node([last_node], skip_name, skip, (b, n))

            shortcuts_add_inputs = []
            # Add shortcut:
            if shortcuts or sh_res:
                sh_sources = self.get_shortcut_source_nodes(b, n, n_blocks, shift)
                for i, (sh_in_node, chan_div) in enumerate(sh_sources):
                    # sh_in_node, chan_div = self.get_shortcut_source_node(b, n, n_blocks, shift)
                    sh_in_chan = int(prev_in_chan / chan_div)
                    if sh_in_node == self.INPUT_NAME and self.bottleneck:
                        special_stride = 2
                    else:
                        special_stride = None
                    if shortcuts:
                        shortcut_block = self.block_type(sh_in_chan, out_chan, stride=special_stride, bias=bias)
                        shortcut_block_name = self.SHORTCUT_BLOCK_NAME.format(b, n, i)
                        self.add_node([sh_in_node], shortcut_block_name, shortcut_block, (b, n))
                        shortcuts_add_inputs.append(shortcut_block_name)
                    # Add skip connection
                    if sh_res:
                        sh_skip = Skip_Block(sh_in_chan, out_chan, stride=special_stride, bias=bias)
                        sh_skip_name = self.SHORTCUT_SKIP_NAME.format(b, n, i)
                        self.add_node([sh_in_node], sh_skip_name, sh_skip, (b, n))
                        shortcuts_add_inputs.append(sh_skip_name)

            # Add addition
            add_block = Add_Block()
            add_name = self.ADD_NAME.format(b, n)
            self.add_node(shortcuts_add_inputs + [skip_name, basic_block_name], add_name, add_block, (b, n))
            in_chan = out_chan
            last_node = add_name

        return last_node

    def add_node(self, in_nodes, node_name, module, pos, **args):
        pos = ResCNFDrawer.get_draw_pos(pos=pos, node_name=node_name)
        sampling_param = sampling_param_generator(self.static_node_proba, node_name)

        self.graph.add_node(node_name, module=module, sampling_param=len(self.sampling_parameters), pos=pos, **args)

        if sampling_param is not None:
            self.sampling_parameters.append(sampling_param)
        else:
            raise RuntimeError('Old version, should be fixed !')
        if isinstance(module, nn.Module):
            self.blocks.append(module)
        else:
            raise RuntimeError('Old version, should be fixed !')


        for input in in_nodes:
            self.graph.add_edge(input, node_name, width_node=node_name)

    def get_shortcut_source_nodes(self, block, depth, max_layers, shift):
        assert block > 0
        sources = []

        source_b = block - 1
        if 's' in shift:
            n = max_layers - depth - 2
            if n < 0:
                if block == 1:
                    sources.append((self.INPUT_NAME, self.scale_factor))
                else:
                    sources.append((self.IDENT_NAME.format(source_b), 2))
            else:
                sources.append((self.ADD_NAME.format(source_b, n), 1))

        if 'l' in shift and depth != 0:
            sources.append((self.ADD_NAME.format(source_b, max_layers - 1 - depth), 1))

        if 'r' in shift and depth != max_layers - 1:
            n = max_layers - depth - 3
            if n < 0:
                if block == 1:
                    sources.append((self.INPUT_NAME, self.scale_factor))
                else:
                    sources.append((self.IDENT_NAME.format(source_b), 2))
            else:
                sources.append((self.ADD_NAME.format(source_b, n), 1))

        return sources


def sampling_param_generator(static_node_proba, node_name):
    if static_node_proba >= 0:
        param_value = 1 if np.random.rand() < static_node_proba else -1
        param_value *= np.inf
        trainable = False
    else:
        param_value = 3
        trainable = True

    return nn.Parameter(torch.Tensor([param_value]), requires_grad=trainable)
