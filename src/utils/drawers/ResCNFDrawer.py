from collections import defaultdict

from src.utils.drawers.Drawer import Drawer


class ResCNFDrawer(Drawer):
    INPUT_NAME = 'Input'
    OUTPUT_NAME = 'Output'
    CLASSIC_BLOCK_NAME = 'CLASSIC'
    SHORTCUT_BLOCK_NAME = 'SHORTCUT'
    CLASSIC_SKIP_NAME = 'SKIP_CL'
    SHORTCUT_SKIP_NAME = 'SKIP_SH'
    ADD_NAME = 'ADD'
    IDENT_NAME = 'IDENT'

    NODE_SIZE = 50
    EDGE_WIDTH = 5.0

    x_max = 0
    y_max = 0
    x_block = defaultdict(int)

    @staticmethod
    def get_draw_pos(node_name=None, source=None, dest=None, pos=None):
        if node_name.startswith(ResCNFDrawer.INPUT_NAME):
            position = pos
        elif node_name.startswith(ResCNFDrawer.OUTPUT_NAME):
            # position = ResCNFDrawer.x_max + 1, ResCNFDrawer.y_max
            if pos[0] % 2 == 0:
                position = ResCNFDrawer.x_block[ResCNFDrawer.x_max] + 2, ResCNFDrawer.y_max
            else:
                position = -1, ResCNFDrawer.y_max
        elif node_name.startswith(ResCNFDrawer.CLASSIC_BLOCK_NAME):
            if pos[0] == 0:

                ResCNFDrawer.x_max = pos[1]
                ResCNFDrawer.x_block[pos[1]] = 2 * pos[1] + 1
                position = ResCNFDrawer.x_block[pos[1]], 0
            elif pos[0] % 2 == 0:
                # ltr
                ResCNFDrawer.y_max = -2 * pos[0]
                position = ResCNFDrawer.x_block[pos[1]], -2 * pos[0]
            else:
                # rtl
                ResCNFDrawer.y_max = -2 * pos[0]
                position = ResCNFDrawer.x_block[ResCNFDrawer.x_max - pos[1]], -2 * pos[0]

        elif node_name.startswith(ResCNFDrawer.SHORTCUT_BLOCK_NAME):
            if pos[0] % 2 == 0:
                position = ResCNFDrawer.x_block[pos[1]] + 1, -2 * pos[0] + 1.5
            else:
                position = ResCNFDrawer.x_block[ResCNFDrawer.x_max - pos[1]] - 1, -2 * pos[0] + 1.5
        elif node_name.startswith(ResCNFDrawer.CLASSIC_SKIP_NAME):
            if pos[0] % 2 == 0:
                position = ResCNFDrawer.x_block[pos[1]], -2 * pos[0] + 1
            else:
                # ResCNFDrawer.y_max = pos[1]
                position = ResCNFDrawer.x_block[ResCNFDrawer.x_max - pos[1]], -2 * pos[0] + 1
        elif node_name.startswith(ResCNFDrawer.SHORTCUT_SKIP_NAME):
            if pos[0] % 2 == 0:
                position = ResCNFDrawer.x_block[pos[1]] + 1.5, -2 * pos[0] + 1.5
            else:
                # ResCNFDrawer.y_max = pos[1]
                position = ResCNFDrawer.x_block[ResCNFDrawer.x_max - pos[1]] - 1.5, -2 * pos[0] + 1.5
        elif node_name.startswith(ResCNFDrawer.ADD_NAME):
            if pos[0] % 2 == 0:
                position = ResCNFDrawer.x_block[pos[1]] + 1, -2 * pos[0]
            else:
                position = ResCNFDrawer.x_block[ResCNFDrawer.x_max - pos[1]] - 1, -2 * pos[0]
        elif node_name.startswith(ResCNFDrawer.IDENT_NAME):
            if pos[0] % 2 == 0:
                position = 0, -2 * pos[0]
            else:
                position = ResCNFDrawer.x_block[ResCNFDrawer.x_max] + 1, -2 * pos[0]
        else:
            raise RuntimeError

        return position

    def draw(self, graph, param_list=None, weights=None, vis_opts=None, vis_win=None, vis_env=None, colormap=None):
        node_filter = lambda n: True
        edge_filter = lambda e: True

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

            def weighter(e):
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
