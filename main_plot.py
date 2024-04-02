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
        try:
            nome_arquivo_array_plot = [folds_raiz + fold]
            array = ler_arquivo_e_transformar_em_array(nome_arquivo_array_plot[0])
            if fold.__contains__("-"):
                A_B = fold.split("-")[0]
            else:
                A_B = fold.split("_")[0]
            nome = 'dataset/Vrp-Set-' + A_B + '/' + fold.replace(".result", "")
            cvrp = CVRP(str(nome))
            cvrp.plot(routes=array, clear_edges=True, stop=False, nome=fold.replace(".result", ""), language="pt")
            cvrp.plot(routes=array, clear_edges=True, stop=False, nome=fold.replace(".result", ""), language="in")
        except:
            print('Erro plot %s',fold)
