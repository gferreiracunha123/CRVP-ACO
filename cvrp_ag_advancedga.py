# Importações necessárias
from cvrp_ag_algorithm import CVRPAlgorithm  # Importa a classe base para o algoritmo genético para o Problema do Caixeiro Viajante (CVRP)
import random  # Importa o módulo random para geração de números aleatórios
import copy  # Importa o módulo copy para realizar cópias profundas de objetos
from heapq import *  # Importa a função heappush do módulo heapq para implementar uma fila de prioridade

# Classe que representa uma população no algoritmo genético avançado (AGA)
class AGAPopulation(object):
    def __init__(self, info, total_iters):
        # Inicializa a população com as informações fornecidas e o número total de iterações
        self.info = info
        self.info.max_route_len = 10  # Define o comprimento máximo da rota
        self.chromosomes = []  # Lista para armazenar os cromossomos da população
        try:
            # Gera soluções aleatórias e melhora-as usando uma abordagem gulosa
            for x in [self.info.steep_improve_solution(self.info.make_random_solution(greedy=True)) for _ in range(800)]:
                heappush(self.chromosomes, (x.cost, x))  # Adiciona a solução à fila de prioridade
        except:
            # Exceção tratada ao imprimir o custo da solução caso ocorra um erro
            print(x.cost, x)

        # Define a melhor solução inicial como a primeira solução na população
        self.best_solution = self.chromosomes[0][1]
        self.iters = 0  # Contador de iterações
        self.total_iters = total_iters  # Número total de iterações
        self.same_route_prob = 0.25  # Probabilidade de realizar uma mutação na mesma rota
        random.seed()  # Inicializa a semente do gerador de números aleatórios

    # Método para avançar a população para a próxima geração
    def step(self):
        replace = 1
        for i in range(12):  # Itera sobre os cromossomos na população
            for j in range(i + 1, 12):  # Itera sobre os cromossomos restantes na população
                ic, jc = self.chromosomes[i][1], self.chromosomes[j][1]  # Seleciona dois cromossomos
                if random.uniform(0, 1) < 0.2:
                    jc = self.chromosomes[random.randrange(10, len(self.chromosomes) - 1)][1]  # Realiza uma troca aleatória
                child = self.biggest_overlap_crossover(ic, jc)  # Realiza o crossover com maior sobreposição
                if random.uniform(0, 1) < 0.95:
                    for _ in range(3):
                        c = self.biggest_overlap_crossover(ic, child)  # Realiza o crossover com maior sobreposição
                        self.info.refresh(c)  # Atualiza as informações do cromossomo
                        if c.cost < child.cost:
                            child = c
                else:
                    for _ in range(3):
                        c = self.simple_random_crossover(ic, child)  # Realiza o crossover simples e aleatório
                        self.info.refresh(c)  # Atualiza as informações do cromossomo
                        if c.cost < child.cost:
                            child = c
                self.info.refresh(child)  # Atualiza as informações do cromossomo
                self.simple_random_mutation(child)  # Realiza a mutação simples e aleatória
                self.info.refresh(child)  # Atualiza as informações do cromossomo
                self.repairing(child)  # Realiza a reparação do cromossomo, se necessário
                self.info.refresh(child)  # Atualiza as informações do cromossomo
                self.info.steep_improve_solution(child)  # Melhora a solução do cromossomo
                self.info.refresh(child)  # Atualiza as informações do cromossomo
                if replace <= len(self.chromosomes):
                    self.chromosomes[-replace] = (self.fitness(child), child)  # Substitui um cromossomo antigo por um novo
                    replace += 1
        heapify(self.chromosomes)  # Reorganiza a fila de prioridade
        self.iters += 1  # Incrementa o contador de iterações
        if self.chromosomes[0][1].cost < self.best_solution.cost:
            self.best_solution = self.chromosomes[0][1]  # Atualiza a melhor solução se necessário
        return self.best_solution

    # Método para calcular o fitness de um cromossomo
    def fitness(self, chromosome):
        penalty = self.penalty(chromosome)  # Calcula a penalidade do cromossomo
        return chromosome.cost + penalty  # Retorna o fitness do cromossomo

    # Método para calcular a penalidade de um cromossomo
    def penalty(self, chromosome):
        penalty_sum = 0  # Inicializa a soma das penalidades
        for route in chromosome.routes:
            penalty_sum += max(0, route.demand - self.info.capacity) ** 2  # Calcula a penalidade para cada rota
        mnv = sum(self.info.demand[i] for i in range(self.info.dimension)) / self.info.capacity  # Calcula a média das demandas
        alpha = self.best_solution.cost / ((1 / (self.iters + 1)) * (self.info.capacity * mnv / 2) ** 2 + 0.00001)  # Calcula o fator alpha
        penalty = alpha * penalty_sum * self.iters / self.total_iters  # Calcula a penalidade total
        chromosome.penalty = penalty  # Armazena a penalidade no cromossomo
        return penalty  # Retorna a penalidade

    # Método para reparar um cromossomo se necessário
    def repairing(self, chromosome):
        routes = chromosome.routes  # Obtém as rotas do cromossomo
        r_max_i = max((i for i in range(len(routes))), key=lambda i: routes[i].demand)  # Encontra o índice da rota com maior demanda
        r_min_i = min((i for i in range(len(routes))), key=lambda i: routes[i].demand)  # Encontra o índice da rota com menor demanda
        if routes[r_max_i].demand > self.info.capacity:  # Verifica se a rota com maior demanda excede a capacidade
            rint = random.randrange(1, len(routes[r_max_i].route) - 1)  # Escolhe aleatoriamente um nó da rota com maior demanda
            routes[r_min_i].append_node(routes[r_max_i].route[rint])  # Move o nó para a rota com menor demanda
            routes[r_max_i].remove_node(routes[r_max_i].route[rint])  # Remove o nó da rota original
            return True  # Indica que uma reparação foi realizada
        return False  # Indica que nenhuma reparação foi necessária

    # Método para realizar o crossover simples e aleatório
    def simple_random_crossover(self, chrom1, chrom2):
        child = copy.deepcopy(chrom1)  # Cria uma cópia profunda do cromossomo pai
        sub_route = chrom2.random_subroute()  # Seleciona uma sub-rota aleatória do cromossomo mãe
        for x in sub_route:
            child.remove_node(x)  # Remove os nós da sub-rota do cromossomo filho
        r_id, n_id = self.best_insertion(child, sub_route)  # Encontra o melhor ponto de inserção para a sub-rota no cromossomo filho
        child.insert_route(r_id, n_id, sub_route)  # Insere a sub-rota no cromossomo filho
        return child  # Retorna o cromossomo filho resultante

    # Método para realizar o crossover de maior sobreposição
    def biggest_overlap_crossover(self, c1, c2):
        child = copy.deepcopy(c1)  # Cria uma cópia profunda do cromossomo pai
        sub_route = c2.random_subroute()  # Seleciona uma sub-rota aleatória do cromossomo mãe
        routes = []
        for x in sub_route:
            child.remove_node(x)  # Remove os nós da sub-rota do cromossomo filho
        for i, route in enumerate(child.routes):  # Itera sobre as rotas do cromossomo filho
            x_min, x_max, y_min, y_max = self.info.bounding_box(route.route)  # Calcula a caixa delimitadora da rota do cromossomo filho
            sx_min, sx_max, sy_min, sy_max = self.info.bounding_box(sub_route)  # Calcula a caixa delimitadora da sub-rota
            x_overlap = max(0, min(x_max, sx_max) - max(x_min, sx_min))  # Calcula a sobreposição na direção x
            y_overlap = max(0, min(y_max, sy_max) - max(y_min, sy_min))  # Calcula a sobreposição na direção y
            heappush(routes, (x_overlap * y_overlap, i))  # Adiciona a sobreposição à fila de prioridade
        top3 = nlargest(6, routes)  # Obtém as 6 maiores sobreposições
        min_i = min((i[1] for i in top3), key=lambda x: child.routes[x].demand)  # Encontra a rota com menor demanda entre as 6 maiores sobreposições
        _, best = self.best_route_insertion(sub_route, child.routes[min_i].route)  # Encontra o melhor ponto de inserção para a sub-rota na rota selecionada
        child.insert_route(min_i, best, sub_route)  # Insere a sub-rota na rota selecionada do cromossomo filho
        return child  # Retorna o cromossomo filho resultante

    # Método para realizar a mutação simples e aleatória
    def simple_random_mutation(self, chromosome):
        r_i = random.randrange(0, len(chromosome.routes))  # Seleciona aleatoriamente uma rota do cromossomo
        while (len(chromosome.routes[r_i].route) == 2):  # Garante que a rota selecionada tenha mais de 2 nós
            r_i = random.randrange(0, len(chromosome.routes))  # Seleciona outra rota aleatória
        c_i = random.randrange(1, len(chromosome.routes[r_i].route) - 1)  # Seleciona aleatoriamente um nó dentro da rota
        node = chromosome.routes[r_i].route[c_i]  # Obtém o nó selecionado
        chromosome.remove_node(node)  # Remove o nó da rota
        if random.uniform(0, 1) < self.same_route_prob:  # Verifica se a mutação deve ser realizada na mesma rota
            _, best = self.best_route_insertion([node], chromosome.routes[r_i].route)  # Encontra o melhor ponto de inserção na mesma rota
            best_i = (r_i, best)  # Índice da rota e do nó de inserção na mesma rota
        else:
            r_r_i = r_i
            while r_i == r_r_i:  # Garante que a rota selecionada para a mutação não seja a mesma
                r_r_i = random.randrange(0, len(chromosome.routes))  # Seleciona uma rota aleatória diferente
            _, best = self.best_route_insertion([node], chromosome.routes[r_r_i].route)  # Encontra o melhor ponto de inserção em uma rota diferente
            best_i = (r_r_i, best)  # Índice da rota e do nó de inserção em outra rota
        chromosome.insert_route(best_i[0], best_i[1], [node])  # Insere o nó na rota selecionada

    # Método para encontrar o melhor ponto de inserção de uma rota
    def best_route_insertion(self, sub_route, route):
        start = sub_route[0]  # Primeiro nó da sub-rota
        end = sub_route[-1]  # Último nó da sub-rota
        best_payoff, best_i = 0, 0  # Inicializa o melhor payoff e o melhor índice de inserção
        dist = self.info.dist  # Obtém a matriz de distâncias
        i = 0
        for i in range(0, len(route) - 1):  # Itera sobre os nós da rota
            init_cost = dist[route[i]][route[i + 1]]  # Custo inicial entre os nós adjacentes na rota
            payoff = init_cost - dist[route[i]][start] - dist[end][route[i + 1]]  # Payoff da inserção da sub-rota entre os nós
            if payoff > best_payoff:
                best_payoff, best_i = payoff, i  # Atualiza o melhor payoff e o melhor índice de inserção
        return best_payoff, i  # Retorna o melhor payoff e o melhor índice de inserção

    # Método para encontrar o melhor índice de rota e índice de nó onde a rota deve ser inserida
    def best_insertion(self, child, sub_route):
        best_payoff, best_rid, best_nid = -1, 0, 0  # Inicializa o melhor payoff, o melhor índice de rota e o melhor índice de nó
        for r_id, route in enumerate(child.routes):  # Itera sobre as rotas do cromossomo filho
            route = route.route  # Obtém a lista de nós da rota
            subopt_best, n_id = self.best_route_insertion(sub_route, route)  # Encontra o melhor ponto de inserção para a sub-rota
            if subopt_best > best_payoff:
                best_payoff, best_rid, best_nid = subopt_best, r_id, n_id  # Atualiza o melhor payoff, o melhor índice de rota e o melhor índice de nó
        return best_rid, best_nid  # Retorna o melhor índice de rota e o melhor índice de nó


# Classe para representar o Algoritmo Genético Avançado para o Problema do Caixeiro Viajante (CVRP)
class CVRPAdvancedGA(CVRPAlgorithm):
    def __init__(self, info, num_populations, total_iters):
        super(CVRPAdvancedGA, self).__init__(info)

        # Inicializa as populações AGA com informações, número de populações e iterações totais
        self.populations = [AGAPopulation(self.info, total_iters) for _ in range(num_populations)]
        self.pop_bests = [0 for _ in range(num_populations)]  # Lista para armazenar as melhores soluções de cada população

    # Método para avançar o algoritmo para a próxima iteração
    def step(self):
        for i, pop in enumerate(self.populations):  # Itera sobre as populações
            self.pop_bests[i] = pop.step()  # Avança cada população para a próxima geração
        self.best_solution = min(self.pop_bests, key=lambda x: x.cost)  # Seleciona a melhor solução entre as populações
        return self.best_solution  # Retorna a melhor solução encontrada
