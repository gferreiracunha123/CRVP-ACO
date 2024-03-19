import os
import random

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform, pdist
import tsplib95


class CVRP:
    _graph = None

    @property
    def graph(self):
        if self._graph is None:
            self._graph = nx.DiGraph()
            self._graph.add_nodes_from(range(self.n))
        return self._graph

    _c = None

    @property
    def c(self):
        if self._c is None:
            self._c = np.round(np.matrix(data=squareform(pdist(self.coord))))
        return self._c

    def __str__(self):
        return self.info['NAME']

    def __init__(self, path: str):

        assert os.path.exists(path), path + ' - arquivo não existe.'
        problem = tsplib95.load(path)
        self.q = problem.capacity
        self.n = problem.dimension
        self.coord = np.array(list(problem.node_coords.values()))
        self.d = np.array(list(problem.demands.values()))

    def generate_random_colors(self,num_colors):
        colors = []
        for _ in range(num_colors):
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            colors.append('#{:02x}{:02x}{:02x}'.format(r, g, b))
        return colors

    def plot(self, routes=None, edges=None, clear_edges=True, stop=True, sleep_time=0.01, nome=None,language="pt"):
        if clear_edges:
            self.graph.clear_edges()
        if routes is not None:
            num_routes = len(routes)
            colors = self.generate_random_colors(num_routes) # Adicione mais cores conforme necessário
            for idx, r in enumerate(routes):
                if len(r) > 1:
                    for i in range(len(r) - 1):
                        self.graph.add_edge(r[i], r[i + 1], color=colors[idx])  # Usar cor diferente para cada rota
                    self.graph.add_edge(r[-1], r[0], color=colors[idx])  # Conectar o último ao primeiro nó
        if edges is not None:
            for i, j in edges:
                self.graph.add_edge(i, j)
        plt.clf()
        edge_colors = nx.get_edge_attributes(self.graph, 'color').values()  # Obtém as cores das arestas
        nx.draw_networkx(self.graph, self.coord, with_labels=False, node_size=20, font_size=5, edge_color=edge_colors)

        # Adicionar legenda para cada rota
        for idx, color in enumerate(colors[:len(routes)]):
            labelRota = "Rota" if "pt" in language else "Route"
            plt.plot([], [], color=color, label=f'{labelRota} {idx + 1}')
        plt.legend()
        # Adicionando título ao gráfico
        plt.title(str(nome).split(".")[0])


        if stop:
            plt.show()
        else:
            plt.draw()
            save = "pt/" if "pt" in language else "in/"
            plt.savefig('images/'+save + str(nome).replace(".vrp", ".png"))
            plt.pause(sleep_time)



    def route_cost(self, routes):
        cost = 0
        for r in routes:
            for i in range(1, len(r)):
                cost += self.c[r[i - 1], r[i]]
            cost += self.c[r[-1], r[0]]
        return cost

    def route_one_cost(self, routesAux):
        cost = 0
        routes = [routesAux]
        for r in routes:
            for i in range(1, len(r)):
                cost += self.c[r[i - 1], r[i]]
            cost += self.c[r[-1], r[0]]
        return cost

    def is_feasible(self, routes):
        if max([self.d[r].sum() for r in routes]) > self.q:
            print("capacidade violada")
            return False
        count = np.zeros(self.n, dtype=int)
        for r in routes:
            for i in r:
                count[i] += 1
        if max(count[1:]) > 1:
            print("cliente vizitado mais de uma vez")
            return False
        if min(count[1:]) < 1:
            print("cliente não vizitado")
            return False
        return True
