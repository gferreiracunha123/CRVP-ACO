from cvrp import CVRP
from os import listdir


def ler_arquivo_e_transformar_em_array(nome_arquivo):
    with open(nome_arquivo, 'r') as file:
        conteudo = file.read().strip()
        array = eval(conteudo)
        string = array.strip('[]')
        sublists = string.split('], [')
        lista_de_listas = []
        for sublist in sublists:
            sublist = sublist.split(', ')
            sublist = [int(element) for element in sublist]
            lista_de_listas.append(sublist)
    return lista_de_listas


if __name__ == '__main__':
    folds_raiz = 'arq_sol/'
    # Lista os diretórios dentro do diretório raiz
    folds = listdir(folds_raiz)
    for fold in folds:
        # Concatena o caminho do diretório raiz com o nome do arquivo
        nome_arquivo_array_plot = [folds_raiz + fold]
        array = ler_arquivo_e_transformar_em_array(nome_arquivo_array_plot[0])
        A_B = 'A' if 'A-' in fold else 'B'
        nome='dataset/Vrp-Set-'+A_B+'/'+fold.replace(".result", "")
        cvrp = CVRP(str(nome))
        cvrp.plot(routes=array, clear_edges=True, stop=False,nome=fold.replace(".result", ""))
