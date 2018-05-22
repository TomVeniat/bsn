import abc


class CostEvaluator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, path_recorder, *args, **kwargs):
        super(CostEvaluator, self).__init__()
        self.path_recorder = path_recorder
        self.costs = None

    @abc.abstractmethod
    def get_cost(self, **kwargs):
        raise NotImplementedError

    def get_costs(self, architectures):
        return [self.get_cost(arch) for arch in architectures]

    def init_costs(self, *args, **kwargs):
        pass

    def get_state(self):
        return {'costs': self.costs}

    def load_state(self, state):
        for key, val in state.items():
            assert hasattr(self, key)
            setattr(self, key, val)

    def new_epoch(self):
        pass
