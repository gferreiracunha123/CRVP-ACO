import argparse
import csv
import statistics
import time
from datetime import datetime
from os import listdir
from os.path import isfile, join
import datetime
import numpy as np

from cvrp import CVRP
from cvrp_ag_advancedga import CVRPAdvancedGA
from cvrp_ag_info import CVRPInfo
from cvrp_ag_runner import CVRPRunnerAg
from heuristicas import Heuristicas


def cria_csv(heuristica_name, arq, nome, qnt_veiculo, total_custo, tempo, interacao, otimo, desvio):
    f = open(arq, 'a', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow([heuristica_name, nome, qnt_veiculo, total_custo, tempo, interacao, otimo, desvio])
    f.close()


def cria_arq_plot(nome, value):
    nome = nome.split('/')[-1]
    try:
        with open(f'arq_sol/{nome}.result', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([value])
    except Exception as e:
        print(f"Erro ao criar o arquivo: {e}")


# 1- apresentar graficos.
# 2- cometarios da funções.
# 3- paramentros main.
# 4- algoritmos para ajuste de parametrôs automaticos.
# 5- rodar os dados no AG e Ant.
# paralelismo aco

def runner(heuristica_name, arq, file, ite: int = None, ants: int = 20, evapor=0.1, k=3, worst=False, elitist=False,
           num_populations=5,
           total_iters=100, AG=False):
    # Inicializa o índice como 0
    index = 0
    # Itera sobre os arquivos fornecidos

    routes = []

    # Verifica se o arquivo é "A-n32-k5.vrp"
    if "" in file:
        # Cria uma instância do problema CVRP
        cvrp = CVRP(str(file))
        # Cria uma instância das heurísticas para o problema CVRP
        heuristicas = Heuristicas(cvrp, plot=False)
        # Imprime uma mensagem indicando o início do processamento do arquivo
        print("Start :" + str(file).split('/')[-1])
        # Registra o tempo de início do processamento
        inicio = time.time()

        # Define o número de iterações baseado na metade do número de nós, se não for especificado
        try:
            if ite == None:
                # olhar concetração de pontos e possivel ??
                # Olhar as demandas ??
                ite, ants = 2, 20
                
                print("Número de iterações:", ite)
                #ite = int(cvrp.n * 0.4)  # 10% do número total de nós para as iterações
                #ants = int(cvrp.n * 0.6)  # 60% do número total de nós para o número de formigas
        except Exception as e:
            # Exibe uma mensagem de erro se houver um problema com o cálculo do número de iterações
            print("ite: " + cvrp.n + " e: " + str(e))

        # Executa o algoritmo genético para gerar a população incial.
        transfomation = None
        if AG:
            cvrp_ag = CVRPRunnerAg(CVRPAdvancedGA(CVRPInfo(file, debug=False), num_populations, total_iters), 1)
            result = cvrp_ag.run()
            # Converte as rotas obtidas para um formato compatível com o algoritmo de colônia de formigas
            list = []
            if result is None:
                resultAux = cvrp_ag.algorithm.best_solution.routes
                for route in resultAux:
                    list.append(route.cast_list_int())
                transfomation = [[element - 1 for element in sublist] for sublist in list]

            if result != None:
                for route in result.routes:
                    list.append(route.cast_list_int())
                transfomation = [[element - 1 for element in sublist] for sublist in list]

        if AG == False or transfomation is not None:
            # Executa o algoritmo de colônia de formigas para otimizar as rotas
            cost, routes, interacao = heuristicas.ant_colony(sol=transfomation, ite=ite, ants=ants, k=k,
                                                             worst=worst,
                                                             elitist=elitist,
                                                             evapor=evapor)
            # Registra o tempo de término do processamento
            fim = time.time()
            # Imprime uma mensagem indicando o fim do processamento do arquivo
            print("Finalizado")

            # Imprime as rotas e seus custos
            for route in routes:
                print("Rotas: ", route, " Custo: ", cvrp.route_one_cost(route))

            # Cria um arquivo CSV com os resultados
            # Converte o timestamp em um objeto datetime
            diferenca_segundos = fim - inicio
            horas = int(diferenca_segundos // 3600)
            minutos = int((diferenca_segundos % 3600) // 60)
            segundos = int(diferenca_segundos % 60)

            # Formatar a diferença de tempo
            tempo_formatado = "{:02}:{:02}:{:02}".format(horas, minutos, segundos)
            cost_result = cvrp.route_cost(routes)
            optimal_value = 0
            if cvrp.optimal_value != None:
                try:
                    round(cvrp.optimal_value, 2)
                except Exception as e:
                    print(e);
            cria_csv(heuristica_name, arq, str(file).split('/')[-1], len(routes), cost_result,
                     tempo_formatado, interacao, optimal_value, statistics.stdev([cost_result, optimal_value]))

def calculate_stop_criteria(cvrp):
    # Calcula a densidade de pontos
    y_array = cvrp.coord[:, 0]  # Considerando que a primeira coluna representa as coordenadas x
    z_array = cvrp.coord[:, 1]

    point_density = cvrp.n / ((max(y_array) - min(y_array)) * (max(z_array) - min(z_array)))

    # Calcula o desvio padrão das demandas
    demand_std = np.std(cvrp.d)

    # Heurística para determinar o número de iterações e formigas
    num_iterations = max(10, min(int(cvrp.n * 0.6), 100))
    num_ants = max(5, min(int(cvrp.n * 0.6), 50))

    # Ajuste com base na densidade de pontos e no desvio padrão das demandas
    if point_density > 0.1:
        num_iterations *= 2
    if demand_std > 10:
        num_ants *= 2

    return num_iterations, num_ants
if __name__ == '__main__':
    # Define o nome do arquivo CSV para armazenar os resultados da análise

    # Define o diretório raiz dos datasets
    folds_raiz = 'dataset/'
    # Lista os diretórios dentro do diretório raiz
    folds = listdir(folds_raiz)
    # Inicializa uma lista vazia para armazenar os arquivos
    files = []
    # Para cada diretório dentro dos diretórios listados
    for fold in folds:
        # Lista os arquivos dentro do diretório
        listA = [f for f in listdir(folds_raiz + fold + "/") if isfile(join(folds_raiz + fold + "/", f))]
        # Concatena o caminho do diretório raiz com o nome do arquivo
        listB = [folds_raiz + fold + "/" + item for item in listA]
        # Adiciona a lista de arquivos resultante à lista de arquivos
        files.append(listB)
    # Inicializa um índice
    index = 0
    # Inicializa uma lista vazia para armazenar os resultados estendidos
    resultado_extend = []
    # Para cada arquivo na lista de arquivos
    for file in files:
        # Estende a lista de resultados estendidos com os arquivos
        resultado_extend.extend(file)
    # Filtra apenas os arquivos que não têm extensão ".sol"
    nova_files = [item for item in resultado_extend if ".sol" not in item]

    # Configuração e análise dos argumentos passados pelo terminal
    parser = argparse.ArgumentParser(description="Descrição do seu programa")
    parser.add_argument("--ite", type=int, default=None, help="Número de iterações")
    parser.add_argument("--ants", type=int, default=20, help="Número de formigas")
    parser.add_argument("--evapor", type=float, default=0.1, help="Taxa de evaporação")
    parser.add_argument("-k", type=int, default=3, help="Parâmetro k")
    parser.add_argument("--worst", action="store_true", help="Indicador para usar a pior formiga")
    parser.add_argument("--elitist", action="store_true", help="Indicador para usar a melhor formiga")
    parser.add_argument("--AG", action="store_true", help="Inicia a popualação inicial com Algoritimo genetico")
    parser.add_argument("--num_populations", type=int, default=5, help="Número de populações AG")
    parser.add_argument("--total_iters", type=int, default=100, help="Total de iterações AG")
    parser.add_argument("--file", type=str, default=100, help="nome do arquivo")
    # Parseia os argumentos da linha de comando
    args = parser.parse_args()
    ag = "AG_ACO" if args.AG else "ACO"
    # Gerar um timestamp atual
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

    # Nome do arquivo com timestamp concatenado
    arq = "analise/analise_Ant_"+ag+"_"+timestamp+".csv"
    # Abre o arquivo CSV em modo de escrita
    f = open(arq, 'w', newline='', encoding='utf-8')
    # Cria um escritor CSV
    w = csv.writer(f)
    # Escreve a linha de cabeçalho no arquivo CSV
    w.writerow(["Algoritmo", "Nome", "qnt Veiculos", "custo", "Tempo", "iteracao", "Valor Otimo", "Desvio padrão"])
    # Fecha o arquivo CSV
    f.close()

    # Chama a função `runner` com os argumentos analisados
    for file in nova_files:
        # Verifica se o arquivo é do tipo ".vrp"
        if not ".vrp" in file:
            # Se não for, pula para o próximo arquivo
           # print("skip: " + file)
            continue
        if  args.file in file:
            runner("+AG", arq, file, args.ite, args.ants, args.evapor, args.k, args.worst, args.elitist,
                   args.num_populations, args.total_iters, True)
