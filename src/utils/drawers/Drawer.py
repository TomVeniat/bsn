import io
import json
import logging

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import visdom
from plotly.graph_objs import *

logger = logging.getLogger(__name__)


class Drawer(object):
    __default_env = None

    def __init__(self, env=None):
        with open('resources/visdom.json') as file:
            draw_params = json.load(file)
        server = draw_params['url']
        port = draw_params['port']
        self.vis = visdom.Visdom(server=server, port=port)
        self.win = None
        self.env = Drawer.__default_env if env is None else env

        logger.info('Init drawer to {}:{}/env/{}'.format(server, port, self.env))

    @staticmethod
    def set_default_env(env):
        Drawer.__default_env = env

    def set_env(self, env):
        self.env = env
        return self

    def draw_weights(self, graph, weights=None, vis_opts=None, vis_win=None, vis_env=None):
        node_filter = lambda n: True
        edge_filter = lambda e: True

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
        if 'width' not in vis_opts:
            vis_opts['width'] = 600
        if 'height' not in vis_opts:
            vis_opts['height'] = 450

        self.win = self.vis.svg(svgstr=img, win=win, opts=vis_opts, env=env)

    def _draw_net(self, graph, filename=None, show_fig=False, normalize=False,
                  nodefilter=None, edgefilter=None, positioner=None, weighter=None, colormap=None):
        plt.close()

        nodes = None
        if nodefilter:
            nodes = [node for node in graph.nodes() if nodefilter(node)]

        edges = None
        if edgefilter:
            edges = [edge for edge in graph.edges() if edgefilter(edge)]

        if positioner is None:
            pos = nx.spring_layout(graph)
        else:
            pos = dict((n, positioner(n)) for n in graph.nodes())

        weights = 1.0
        if weighter is not None:
            weights = [weighter(e) for e in edges]

        weights = np.array(weights)
        w_min = weights.min()
        w_max = weights.max()
        if normalize and w_min != w_max:
            weights = np.log(weights + 1e-5)
            weights = (weights - w_min) * 1.0 / (w_max - w_min) + 2

        v_min = w_min - .1

        if colormap is None:
            colormap = plt.cm.YlGnBu

        nx.draw_networkx_nodes(graph, nodelist=nodes, pos=pos, node_size=self.NODE_SIZE, node_color='red')
        res = nx.draw_networkx_edges(graph, edgelist=edges, pos=pos, width=self.EDGE_WIDTH, arrows=False,
                                     edge_color=weights,
                                     edge_cmap=colormap, edge_vmin=v_min)

        plt.colorbar(res)

        if show_fig:
            plt.show()
        if filename is not None:
            plt.savefig(filename, format='svg')

        img_data = io.StringIO()
        plt.savefig(img_data, format='svg')

        return img_data.getvalue()

    def scatter(self, x, y, opts, vis_win):
        points = []
        labels = []
        legend = []

        for i, (name, abs) in enumerate(x.items()):
            ord = 0 if y is None else y[name]

            points.append((abs, ord))
            labels.append(i + 1)
            legend.append(name)

        points = np.asarray(points)

        vis_opts = dict(
            legend=legend,
            markersize=5,
        )
        vis_opts.update(opts)
        self.vis.scatter(points, labels, win=vis_win, opts=vis_opts, env=self.env)
