from past.builtins import raw_input

from cvrp_ag_info import CVRPInfo
from cvrp_ag_advancedga import CVRPAdvancedGA
import os
import time
import signal


class CVRPRunnerAg(object):

    def __init__(self, algorithm, iterations):
        self.algorithm = algorithm
        self.print_cycle = 10
        self.num_iter = iterations
        self.iter = 0

    def run(self):
        aux=None
        self.start_time = time.time()
        while self.iter < self.num_iter:
            try:
                best = self.algorithm.step()
                self.best = best
                aux = self.best
                # aux = best
            except Exception as e:
                print(e)
                break
        return aux