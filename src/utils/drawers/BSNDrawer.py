import numpy as np

from src.utils.drawers.Drawer import Drawer


class BSNDrawer(Drawer):
    NODE_SIZE = 100
    EDGE_WIDTH = 5

    @staticmethod
    def get_draw_pos(source=None, dest=None, pos=None):
        if pos is not None:
            return pos[0], -pos[1]
        if source is not None and dest is not None:
            src_l, src_s = source
            dst_l, dst_s = dest
            return np.mean([src_l, src_l, dst_l]), -np.mean([src_s, src_s, dst_s])

    def draw_weights(self, graph, weights=None, vis_opts=None, vis_win=None, vis_env=None):
        node_filter = lambda n: n.startswith('A')
        edge_filter = lambda e: 'width_node' in graph.get_edge_data(*e)

        if weights is None:
            def weighter(e):
                width_node = graph.node[graph.get_edge_data(*e)['width_node']]
                if 'cost' in width_node.keys():
                    return width_node['cost']
                else:
                    return 1
        else:
            raise RuntimeError("draw_weights can't have weights param")

        def positioner(n):
            if 'pos' not in graph.node[n]:
                return None
            return graph.node[n]['pos']

        img = self._draw_net(graph, nodefilter=node_filter, edgefilter=edge_filter,
                             positioner=positioner, weighter=weighter)

        env = vis_env if vis_env is not None else self.env
        win = vis_win if vis_win is not None else self.win
        self.win = self.vis.svg(svgstr=img, win=win, opts=vis_opts, env=env)

    def draw(self, graph, param_list=None, weights=None, vis_opts=None, vis_win=None, vis_env=None, colormap=None):
        node_filter = lambda n: True  # todo Should be removed
        edge_filter = lambda e: True  # todo Should be removed

        if param_list is not None:
            def weighter(e):
                width_node = graph.node[graph.get_edge_data(*e)['width_node']]
                return param_list[width_node['sampling_param']].data[0]
        elif weights is None:
            def weighter(e):
                width_node = graph.node[graph.get_edge_data(*e)['width_node']]
                if 'sampling_val' in width_node and width_node['sampling_val'] is not None:
                    return width_node['sampling_val']
                else:
                    return width_node['sampling_param'].data[0]
        elif type(weights) is float:
            def weighter(_):
                return weights
        else:
            def weighter(e):
                return weights[graph.get_edge_data(*e)['width_node']]

        def positioner(n):
            if 'pos' not in graph.node[n]:
                return None
            return graph.node[n]['pos']

        img = self._draw_net(graph, nodefilter=node_filter, edgefilter=edge_filter,
                             positioner=positioner, weighter=weighter, colormap=colormap)

        env = vis_env if vis_env is not None else self.env
        win = vis_win if vis_win is not None else self.win

        self.win = self.vis.svg(svgstr=img, win=win, opts=vis_opts, env=env)
