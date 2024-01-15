import csv
import time
from os import listdir
from os.path import isfile, join

from cvrp import CVRP
from heuristicas import Heuristicas


def cria_csv(inicio, nome, qnt_veiculo, total_custo, tempo):
    f = open('dataset/analise/analise_Ant.csv', 'a', newline='', encoding='utf-8')

    # 2. cria o objeto de gravação
    w = csv.writer(f)

    # 3. grava as linhas
    if inicio:
        w.writerow(["Nome", "qnt Veiculos ", "custo","Tempo"])
    w.writerow([nome, qnt_veiculo, total_custo, tempo])

    # Recomendado: feche o arquivo
    f.close()


if __name__ == '__main__':

    open('dataset/analise/analise_Ant.csv', 'w', newline='', encoding='utf-8')
    raiz = 'dataset/Vrp-Set-A/'
    files = [f for f in listdir(raiz) if isfile(join(raiz, f))]
    index = 0
    for file in files:
        if not ".vrp" in file:
            print("skip: " + file)
            continue

        cvrp = CVRP(str('dataset/Vrp-Set-A/') + file)
        heuristicas = Heuristicas(cvrp, plot=False)
        print("Start" + file)
        inicio = time.time()
        cost, routes = heuristicas.ant_colony(ite=20, ants=20,k=3, worst=True, elitist=True,
                                              evapor=0.5)
        fim = time.time()
        print("Finalizado")

        for route in routes:
            print("Rotas: ", route, " Custo: ", cvrp.route_one_cost(route))

        cria_csv(index == 0, file, len(routes), cvrp.route_cost(routes), str(fim - inicio))
        index = index + 1
        print("Custo total:", cvrp.route_cost(routes))
        #cvrp.plot(routes=routes)
