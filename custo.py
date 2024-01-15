import math

# Dados dos nós
import numpy as np
from scipy.spatial.distance import squareform, pdist

node_coords = {
    1: (82, 76),
    2: (96, 44),
    3: (50, 5),
    4: (49, 8),
    5: (13, 7),
    6: (29, 89),
    7: (58, 30),
    8: (84, 39),
    9: (14, 24),
    10: (2, 39),
    11: (3, 82),
    12: (5, 10),
    13: (98, 52),
    14: (84, 25),
    15: (61, 59),
    16: (1, 65),
    17: (88, 51),
    18: (91, 2),
    19: (19, 32),
    20: (93, 3),
    21: (50, 93),
    22: (98, 14),
    23: (5, 42),
    24: (42, 9),
    25: (61, 62),
    26: (9, 97),
    27: (80, 55),
    28: (57, 69),
    29: (23, 15),
    30: (20, 70),
    31: (85, 60),
    32: (98, 5)
}

# Rota 1: 21 31 19 17 13 7 26
routes = [[0, 31, 19, 17, 21, 13, 7, 26, 29, 0], [0, 5, 25, 10, 15, 9, 22, 18, 8, 6, 0], [0, 1, 12, 16, 30, 24, 14, 0], [0, 3, 2, 23, 28, 4, 11, 20, 0], [0, 27, 0]]

# Função para calcular o custo total de uma rota


def calculate_route_cost(r, matrix):
    cost = 0
    for i in range(1, len(r)):
        cost += matrix[r[i - 1], r[i]]
    return cost


node_array_float = [[float(x) for x in coords] for coords in node_coords.values()]
matrix = np.round(np.matrix(data=squareform(pdist(node_array_float))))

# Calcula o custo total somando os custos de cada rota
total_total = 0

for rota in routes:
    total_total_total=calculate_route_cost(rota, matrix)
    total_total = total_total + total_total_total
    print("Custo tota:", total_total_total)
print("Custo total da solução:", total_total)
