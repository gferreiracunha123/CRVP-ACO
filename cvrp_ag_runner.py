from past.builtins import raw_input

from cvrp_ag_info import CVRPInfo
from cvrp_ag_advancedga import CVRPAdvancedGA
import os
import time
import signal


class CVRPRunner(object):

    def __init__(self, algorithm, iterations):
        self.algorithm = algorithm
        self.print_cycle = 10
        self.num_iter = iterations
        self.iter = 0
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, signal, frame):
        handling = True
        while handling:
            print(
                "Iter:{0}\nPath:{1}\nWhat do? E for exec(), V for visualise, C to continue, S to save, X to exit".format(
                    self.iter, self.best))
            c = raw_input()
            if c == "E":
                print("exec:")
                exec(raw_input())
            if c == "S":
                self.write_to_file("best-solution-{0}.part".format(self.iter))
            if c == "C":
                handling = False
            if c == "V":
                self.algorithm.info.visualise(self.best).show()
            elif c == "X":
                exit(0)

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
            # best = self.algorithm.step()
            # self.best = best
        return aux

    def write_to_file(self, file_name):
        text = os.linesep.join(["login cm13558 65195",
                                "name Callum Mann",
                                "algorithm Advanced GA",
                                "cost " + str(self.algorithm.best_solution.cost),
                                str(self.algorithm.best_solution)])
        with open(file_name, "w") as f:
            f.write(text)
