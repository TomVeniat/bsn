from src.interfaces.CostEvaluator import CostEvaluator


class EdgeCostEvaluator(CostEvaluator):

    def get_cost(self, architectures):
        costs = self.costs.unsqueeze(1).expand_as(architectures)
        costs = architectures * costs
        return costs.sum(0)

    @property
    def total_cost(self):
        return self.costs.sum().item()
