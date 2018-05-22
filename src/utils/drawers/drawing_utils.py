from matplotlib import colors

from src.implem.BudgetedSuperNetwork import BudgetedSuperNetwork
from src.implem.ResCNF import ResCNF
from src.utils.drawers.BSNDrawer import BSNDrawer
from src.utils.drawers.ResCNFDrawer import ResCNFDrawer


def get_drawer(model, env):
    if isinstance(model, BudgetedSuperNetwork):
        drawer = BSNDrawer
    elif isinstance(model, ResCNF):
        drawer = ResCNFDrawer
    else:
        raise ValueError('Unknown model: {}'.format(type(model)))

    return drawer(env=env)


def get_colormap(n):
    base = ['b', 'y', 'g', 'r', 'c', 'm', 'k']
    base *= int(n / len(base)) + 1

    cmap = colors.ListedColormap(base[:n], N=n)
    return cmap
