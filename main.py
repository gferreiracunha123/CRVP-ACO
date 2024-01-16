import csv
import json
import threading
import time
from os import listdir
from os.path import isfile, join

from cvrp import CVRP
from heuristicas import Heuristicas


def cria_csv(arq, inicio, nome, qnt_veiculo, total_custo, tempo):
    f = open(arq, 'a', newline='', encoding='utf-8')

    # 2. cria o objeto de gravação
    w = csv.writer(f)

    # 3. grava as linhas
    if inicio:
        w.writerow(["Nome", "qnt Veiculos ", "custo", "Tempo"])
    w.writerow([nome, qnt_veiculo, total_custo, tempo])

    # Recomendado: feche o arquivo
    f.close()


def cria_arq_plot(nome, value):
    nome = nome.split('/')[-1]
    try:
        with open(f'arq_sol/{nome}.result', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow([value])
    except Exception as e:
        print(f"Erro ao criar o arquivo: {e}")


def dividir_e_executar(arq, lista, num_threads):
    tamanho_sublista = len(lista) // num_threads
    threads = []
    for i in range(num_threads):
        inicio = i * tamanho_sublista
        fim = (i + 1) * tamanho_sublista if i < num_threads - 1 else None
        sublista = lista[inicio:fim]

        thread = threading.Thread(target=processar_sublista, args=(arq, sublista,))
        thread.start()
        threads.append(thread)

    # for thread in threads:
    # thread.join()


def processar_sublista(arq, files):
    index = 0
    for file in files:
        if not ".vrp" in file:
            print("skip: " + file)
            continue
        routes = []
        try:
            cvrp = CVRP(str(file))
            heuristicas = Heuristicas(cvrp, plot=False)
            print("Start :" + str(file).split('/')[-1])
            inicio = time.time()
            ite = 20
            try:
                ite = int(cvrp.n / 2)
            except Exception as e:
                print("ite: " + cvrp.n + " e: " + e)
            cost, routes = heuristicas.ant_colony(ite=ite, ants=ite, k=3, worst=True, elitist=True,
                                                  evapor=0.5)
            fim = time.time()
            print("Finalizado")

            for route in routes:
                print("Rotas: ", route, " Custo: ", cvrp.route_one_cost(route))

            cria_csv(arq, index == 0, str(file).split('/')[-1], len(routes), cvrp.route_cost(routes), str(fim - inicio))
            index = index + 1
            print("Custo total:", cvrp.route_cost(routes))
        except Exception as e:
            print(f"Erro ao criar rotas: {e}, {file}")
            cria_arq_plot(file + ".error", str(e))
        try:
            cria_arq_plot(file, str(routes))
        except Exception as e:
            print(f"Erro ao criar o cria_arq_plot: {e}, {routes}")


if __name__ == '__main__':
    arq = 'analise/analise_Ant.csv'
    open(arq, 'w', newline='', encoding='utf-8')
    folds_raiz = 'dataset/'
    folds = listdir(folds_raiz)
    files = []
    for fold in folds:
        listA = [f for f in listdir(folds_raiz + fold + "/") if isfile(join(folds_raiz + fold + "/", f))]
        listB = [folds_raiz + fold + "/" + item for item in listA]
        files.append(listB)
    index = 0
    resultado_extend = []
    for file in files:
        resultado_extend.extend(file)
    nova_files = [item for item in resultado_extend if ".sol" not in item]
    # Número de threads desejadas
    num_threads = 15
    # Dividir e executar
    dividir_e_executar(arq, nova_files, num_threads)
