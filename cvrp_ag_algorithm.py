from cvrp_ag_info import CVRPInfo

class CVRPAlgorithm(object):
    def __init__(self, info):
        self.info = info
        self.best_paths = []
        self.best_solution = None

    def random_path(self):
        pass
