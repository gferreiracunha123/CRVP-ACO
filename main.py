import argparse
import csv
import time
from os import listdir
from os.path import isfile, join

from cvrp import CVRP
from cvrp_ag_advancedga import CVRPAdvancedGA
from cvrp_ag_info import CVRPInfo
from cvrp_ag_runner import CVRPRunnerAg
from heuristicas import Heuristicas


def cria_csv(arq, inicio, nome, qnt_veiculo, total_custo, tempo):
    f = open(arq, 'a', newline='', encoding='utf-8')
    w = csv.writer(f)
    w.writerow([nome, qnt_veiculo, total_custo, tempo])
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

def runner(arq, files, ite: int = None, ants: int = 20, evapor=0.1, k=3, worst=False, elitist=False, num_populations=5,
           total_iters=100):
    # Inicializa o índice como 0
    index = 0
    # Itera sobre os arquivos fornecidos
    for file in files:
        # Verifica se o arquivo é do tipo ".vrp"
        if not ".vrp" in file:
            # Se não for, pula para o próximo arquivo
            print("skip: " + file)
            continue
        # Lista para armazenar as rotas
        routes = []

        # Verifica se o arquivo é "A-n32-k5.vrp"
        if "A-n32-k5.vrp" in file:
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
                    ite = int(cvrp.n / 2)
            except Exception as e:
                # Exibe uma mensagem de erro se houver um problema com o cálculo do número de iterações
                print("ite: " + cvrp.n + " e: " + str(e))

            # Executa o algoritmo genético para gerar a população incial.
            cvrp_ag = CVRPRunnerAg(CVRPAdvancedGA(CVRPInfo(file, debug=False), num_populations, total_iters),
                                   1)
            result = cvrp_ag.run()
            # Converte as rotas obtidas para um formato compatível com o algoritmo de colônia de formigas
            list = []
            for route in result.routes:
                list.append(route.cast_list_int())
            transfomation = [[element - 1 for element in sublist] for sublist in list]

            # Executa o algoritmo de colônia de formigas para otimizar as rotas
            cost, routes = heuristicas.ant_colony(transfomation, result.cost, ite=ite, ants=ants, k=k, worst=worst,
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
            cria_csv(arq, index == 0, str(file).split('/')[-1], len(routes), cvrp.route_cost(routes), str(fim - inicio))
            index = index + 1
            # Imprime o custo total das rotas
            print("Custo total:", cvrp.route_cost(routes))

        # Tenta criar um arquivo de plot para as rotas, se falhar, exibe uma mensagem de erro
        try:
            cria_arq_plot(file, str(routes))
        except Exception as e:
            print(f"Erro ao criar o cria_arq_plot: {e}, {routes}")


if __name__ == '__main__':
    # Define o nome do arquivo CSV para armazenar os resultados da análise
    arq = 'analise/analise_Ant.csv'
    # Abre o arquivo CSV em modo de escrita
    f = open(arq, 'w', newline='', encoding='utf-8')
    # Cria um escritor CSV
    w = csv.writer(f)
    # Escreve a linha de cabeçalho no arquivo CSV
    w.writerow(["Nome", "qnt Veiculos ", "custo", "Tempo"])
    # Fecha o arquivo CSV
    f.close()

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
    parser.add_argument("--ite", type=int, default=20, help="Número de iterações")
    parser.add_argument("--ants", type=int, default=20, help="Número de formigas")
    parser.add_argument("--evapor", type=float, default=0.1, help="Taxa de evaporação")
    parser.add_argument("-k", type=int, default=3, help="Parâmetro k")
    parser.add_argument("--worst", action="store_true", help="Indicador para usar a pior formiga")
    parser.add_argument("--elitist", action="store_true", help="Indicador para usar a melhor formiga")
    parser.add_argument("--num_populations", type=int, default=5, help="Número de populações AG")
    parser.add_argument("--total_iters", type=int, default=100, help="Total de iterações AG")

    # Parseia os argumentos da linha de comando
    args = parser.parse_args()

    # Chama a função `runner` com os argumentos analisados
    runner(arq, nova_files, args.ite, args.ants, args.evapor, args.k, args.worst, args.elitist, args.num_populations,
           args.total_iters)
