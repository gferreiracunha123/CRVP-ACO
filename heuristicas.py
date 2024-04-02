from builtins import reversed
import numpy as np
import random as rd
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import time

import tsp
from cvrp import CVRP


def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.4f sec' % (f.__name__, args, kw, te - ts))
        return result

    return timed


def progress(done, total, text: str):
    x = int(round(40.0 * done / total))
    print(f"\r{text}: |{'█' * x}{'-' * (40 - x)}|", end='')
    if done == total:
        print()
    pass


class Heuristicas:

    def __init__(self, cvrp: CVRP, plot=False):

        self.cvrp = cvrp
        self.plot = plot
        self.tabu_list = None
        pass

    def intra_route(self, route, cost=0):
        chg = False
        for r in route:
            imp = True
            while imp:
                imp = tsp.two_opt(r, self.cvrp.c)
                if not imp:
                    imp = tsp.three_opt(r, self.cvrp.c)
                if imp:
                    chg = True
        if chg:
            cost = self.cvrp.route_cost(route)
        assert self.cvrp.is_feasible(route)
        return chg, cost

    def _arg_best_insection(self, route, v):
        c = self.cvrp.c
        n = len(route)
        min_arg = n
        min_val = c[route[-1], v] + c[v, route[0]] - c[route[-1], route[0]]
        for i in range(1, n):
            d = c[route[i - 1], v] + c[v, route[i]] - c[route[i - 1], route[i]]
            if d < min_val:
                min_val = d
                min_arg = i
        return min_arg, min_val

    def replace(self, route, cost=0):
        q = self.cvrp.q
        c = self.cvrp.c
        d = self.cvrp.d
        chg = False

        # Calcula a carga de cada rota
        load = np.array([d[r].sum() for r in route])

        # Loop principal para realizar a operação de substituição
        while True:
            imp = False  # Flag para indicar se houve uma melhoria
            for a, ra in enumerate(route):
                for i, vi in enumerate(ra[1:], start=1):  # Começa de 1 para ignorar o primeiro nó
                    rem_cost = c[ra[i - 1], ra[(i + 1) % len(ra)]] - c[ra[i - 1], vi] - c[vi, ra[(i + 1) % len(ra)]]
                    if rem_cost > -1e-3:
                        continue

                    # Inicializa os valores mínimos
                    min_val = np.inf
                    min_arg = None

                    # Procura por uma rota onde o nó vi possa ser inserido
                    for b, rb in enumerate(route):
                        if load[b] + d[vi] <= q and a != b:
                            insert_pos, add_cost = self._arg_best_insection(rb, vi)
                            if add_cost < min_val and add_cost + rem_cost < -1e-3:
                                # Verifica se a operação é tabu
                                if self.tabu_list is not None:
                                    if self._is_tabu(set(ra) - set([vi]), cost + add_cost + rem_cost) or self._is_tabu(
                                            rb + [vi], cost + add_cost + rem_cost):
                                        continue
                                min_val = add_cost
                                min_arg = b, insert_pos
                                if min_val < 1e-3:
                                    break

                    # Se encontrou uma posição de inserção válida
                    if min_arg is not None and min_val + rem_cost < -1e-3:
                        del ra[i]  # Remove o nó da rota atual
                        load[a] -= d[vi]  # Atualiza a carga da rota atual
                        route[min_arg[0]].insert(min_arg[1], vi)  # Insere o nó na nova posição
                        load[min_arg[0]] += d[vi]  # Atualiza a carga da nova rota
                        chg = imp = True
                        cost += min_val + rem_cost  # Atualiza o custo
                        break

            # Se não houve melhorias, sai do loop principal
            if not imp:
                break

        assert self.cvrp.is_feasible(route)
        return chg, cost

    def swap(self, route, cost=0):
        q = self.cvrp.q
        c = self.cvrp.c
        d = self.cvrp.d
        chg = False

        # Calcula a carga de cada rota
        load = np.array([d[r].sum() for r in route])

        # Função auxiliar para calcular a distância entre dois vértices
        def calculate_delta(a, i, b, j):
            nonlocal load
            vi = route[a][i]
            vj = route[b][j]
            delta = (c[route[a][i - 1], vj] + c[vj, route[a][(i + 1) % len(route[a])]] -
                     c[route[a][i - 1], vi] - c[vi, route[a][(i + 1) % len(route[a])]] +
                     c[route[b][j - 1], vi] + c[vi, route[b][(j + 1) % len(route[b])]] -
                     c[route[b][j - 1], vj] - c[vj, route[b][(j + 1) % len(route[b])]])
            return delta

        # Loop principal para realizar a operação de troca
        while True:
            imp = False
            for a in range(1, len(route)):
                for i in range(1, len(route[a])):
                    vi = route[a][i]
                    for b in range(a):
                        for j in range(1, len(route[b])):
                            vj = route[b][j]
                            delta = calculate_delta(a, i, b, j)
                            if delta < -1e-3 and load[a] + d[vj] - d[vi] <= q and load[b] + d[vi] - d[vj] <= q:
                                if self.tabu_list is not None:
                                    if self._is_tabu(route[a], cost + delta) or self._is_tabu(route[b], cost + delta):
                                        continue
                                route[a][i], route[b][j] = vj, vi
                                load[a] += d[vj] - d[vi]
                                load[b] += d[vi] - d[vj]
                                chg = imp = True
                                cost += delta
                                break
                        if imp:
                            break
                    if imp:
                        break
                if imp:
                    break

            # Se não houve melhorias, sai do loop principal
            if not imp:
                break

        assert self.cvrp.is_feasible(route)
        return chg, cost

    def two_opt_star(self, route, cost=0):
        q = self.cvrp.q
        c = self.cvrp.c
        d = self.cvrp.d
        chg = False

        while True:
            imp = False

            for a in range(1, len(route)):
                ra = route[a]

                if len(ra) < 3:
                    continue

                for i in range(1, len(ra)):
                    vi = ra[i]
                    vni = ra[(i + 1) % len(ra)]

                    for b in range(a):
                        rb = route[b]

                        if len(rb) < 3:
                            continue

                        for j in range(1, len(rb)):
                            vj = rb[j]
                            vnj = rb[(j + 1) % len(rb)]

                            delta = c[vj, vni] + c[vi, vnj] - c[vi, vni] - c[vj, vnj]

                            if delta < -1e-3:
                                if sum(d[ra[0:i + 1]]) + sum(d[rb[j + 1:]]) <= q and sum(d[rb[0:j + 1]]) + sum(
                                        d[ra[i + 1:]]) <= q:
                                    # Adaptação para o tabu
                                    if self.tabu_list is not None:
                                        if self._is_tabu(ra[0:i + 1] + rb[j + 1:], cost + delta) or self._is_tabu(
                                                rb[0:j + 1] + ra[i + 1:], cost + delta):
                                            continue

                                    na = ra[0:i + 1] + rb[j + 1:]
                                    nb = rb[0:j + 1] + ra[i + 1:]

                                    ra.clear()
                                    ra.extend(na)
                                    rb.clear()
                                    rb.extend(nb)

                                    chg = imp = True
                                    cost += delta

                                    if self.plot:
                                        self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)

                                    break

                            delta = c[vnj, vni] + c[vi, vj] - c[vi, vni] - c[vj, vnj]

                            if delta < -1e-3:
                                if sum(d[ra[:i + 1]]) + sum(d[rb[:j + 1]]) <= q and sum(d[rb[j + 1:]]) + sum(
                                        d[ra[i + 1:]]) <= q:
                                    # Adaptação para o tabu
                                    if self.tabu_list is not None:
                                        if self._is_tabu(ra[:i + 1] + rb[j:0:-1], cost + delta) or self._is_tabu(
                                                [0] + ra[:i:-1] + rb[j + 1:], cost + delta):
                                            continue

                                    na = ra[:i + 1] + rb[j:0:-1]
                                    nb = [0] + ra[:i:-1] + rb[j + 1:]

                                    ra.clear()
                                    ra.extend(na)
                                    rb.clear()
                                    rb.extend(nb)

                                    chg = imp = True
                                    cost += delta

                                    if self.plot:
                                        self.cvrp.plot(routes=route + [ra], clear_edges=True, stop=False)

                                    break

                        if imp:
                            break

                    if imp:
                        break

            if not imp:
                break

        assert self.cvrp.is_feasible(route)
        return chg, cost

    def VND(self, sol, cost=None):
        """
        Variable Neighborhood Descent
        :param sol: Solução (lista de listas)
        :param cost: Custo atual da solução
        :return: tupla (custo, solução)
        """
        if cost is None:
            cost = self.cvrp.route_cost(sol)

        neighborhoods = [self.swap, self.replace, self.two_opt_star, self.intra_route]
       # neighborhoods = [self.replace]

        while True:
            np.random.shuffle(neighborhoods)
            imp = False

            for neighborhood in neighborhoods:
                imp, cost = neighborhood(sol, cost)
                if imp:
                    break

            if not imp:
                break

        # eliminar rotas vazias
        sol = [route for route in sol if len(route) > 1]

        assert self.cvrp.is_feasible(sol)
        assert cost == self.cvrp.route_cost(sol)
        return cost, sol

    def _shake(self, sol, cost, k=1, tenure=0):
        """
        Perturba a solução destruindo rotas e reconstruindo com algoritmo de saving
        As rotas destruídas se tornam tabus e não poderão ser reconstruídas

        :param sol: Solução (lista de listas)
        :param k: número de rotas a serem destruídas
        :param tenure: tamanho máximo da lista tabu, usado quando self.tabu_list não é None
        :return: tupla (custo, solução)
        """
        # Seleciona k rotas para destruição
        destruct_list = rd.sample(range(len(sol)), k)
        v = []

        # Clientes sem rota
        for r in sorted(destruct_list, reverse=True):
            # Destrói rotas
            v.extend(sol[r][1:])
            if self.tabu_list is not None:
                # Cria tabu
                self.tabu_list.append((set(sol[r]), cost))
                if len(self.tabu_list) > tenure:
                    del self.tabu_list[0]
            del sol[r]

        # Cria rotas triviais para os clientes sem rota
        sol.extend([[0, i] for i in v])
        cost = self.cvrp.route_cost(sol)
        return cost, sol

    def _is_tabu(self, r: [list, set], cost: float):
        if r is not set:
            r = set(r)
        for s, c in self.tabu_list:
            if s == r and c <= cost:
                return True
        return False

    def tabu_search(self, ite: int, k: int, tenure: int, reset_factor=1.05, cost: int = None,
                    sol=None):

        self.tabu_list = []
        best_cost = cost
        best_sol = sol
        current_sol = deepcopy(best_sol)
        current_cost = best_cost
        for i in range(ite):
            current_cost, current_sol = self._shake(current_sol, current_cost, k, tenure)
            current_cost, current_sol = self.VND(current_sol, current_cost)
            if best_cost > current_cost:
                self.tabu_list.clear()
                best_cost = current_cost
                best_sol = deepcopy(current_sol)
            elif best_cost * reset_factor < current_cost:
                self.tabu_list.clear()
                current_sol = deepcopy(best_sol)
        self.tabu_list = None
        return best_cost, best_sol

    def _ant_run(self, trail):

        n = self.cvrp.n
        d = self.cvrp.d
        q = self.cvrp.q
        c = self.cvrp.c
        sol = []

        maxc = c.max()

        visited = np.zeros([n], dtype=bool)

        cont = 1
        qOtmizador = q * 0.5
        while cont < n:
            path = [0]
            v = 0
            load = float(0)
            while True:
                can = [i for i in range(n) if not visited[i] and load + d[i] <= q and v != i]
                if len(can) == 0:
                    break
                weight = np.array([max(trail[v, i], self._min_trail) for i in can])

                # heuristica de minimazão,  prioriza a menor rota
                heu = np.array([(maxc - c[v, i]) / maxc for i in can])
                if v != 0:
                    # se carga for menor de 50% prioriza as rotas mais longe do deposito
                    if load < qOtmizador:
                        heu *= np.array([2 if c[0, i] > c[0, v] else 1 for i in can])
                    else:
                        heu *= np.array([1 if c[0, i] < c[0, v] else 2 for i in can])

                heu /= heu.max()
                weight /= weight.max()  # normalizar

                weight = weight * heu
                # algoritimo de roleta
                v = rd.choices(can, weights=weight)[0]
                if v == 0:
                    break
                else:
                    path.append(v)
                    load += d[v]
                    visited[v] = True
                    cont += 1
            sol.append(path)

        return sol

    _min_trail = 0.001

    def _reinforcement(self, sol, valor, trail):
        c = self.cvrp.c
        trail_copy = np.copy(trail)

        for route in sol:
            start_node = route[0]
            for i, node in enumerate(route[:-1]):
                next_node = route[i + 1]
                if c[node, next_node] < c[route[-1], start_node]:
                    trail_copy[node, next_node] += valor
                else:
                    trail_copy[next_node, node] += valor

        return trail_copy

    def ant_colony(self, sol, ite: int, ants: int, evapor=0.1, k=1, worst=False, elitist=False):
        print("Processando...")
        n = self.cvrp.n
        trail = np.zeros(shape=[n, n], dtype=float)
        trailLocal = np.zeros(shape=[n, n], dtype=float)
        best_route = None
        best_cost = np.inf
        listBests = []
        cost = None
        num_intracao = 0
        if sol is not None:
            trail = (1 - evapor) * trail
            self._reinforcement(sol, 1, trail)
        paradaInterna = int(ants * 0.4)
        paradaExterna = int(ite * 0.7)
        for i in range(ite):
            lista = []

            if self.force_stop(listBests, best_cost, paradaExterna):
                break

            with ThreadPoolExecutor(max_workers=ants) as executor:
                futures = [executor.submit(self._ant_run, trail) for _ in range(ants)]

                for future in tqdm(as_completed(futures), total=ants, desc=f'Turno: {i + 1}/{ite}={best_cost}'):
                    sol = future.result()
                    cost = self.cvrp.route_cost(sol)
                    cost, sol = self.tabu_search(2, k, 10, 1.01, cost, sol)
                    lista.append((cost, sol))
                    if self.force_stop(listBests, cost, paradaInterna):
                        break

                    if cost < best_cost:
                        best_cost = cost
                        best_route = deepcopy(sol)
                        listBests.append(best_cost)

            listBests.append(best_cost)
            # evaporação
            trail = (1 - evapor) * trail

            # reforço
            if worst:
                # pega pior solução e atualiza a lista de feromonio com -1
                cost, sol = max(lista)
                self._reinforcement(sol, -1, trail)

            if elitist:
                # pega melhor solução e atualiza a lista de feromonio com 1
                self._reinforcement(best_route, 1, trail)

            lista.sort()
            delta = k
            for cost, sol in lista[:k]:
                self._reinforcement(sol, delta, trail)
                delta -= 1

            num_intracao = i

        return best_cost, best_route, num_intracao

    def force_stop(self, lista, cost, stop):
        if lista.count(cost) >= stop:
            return True
        return False
