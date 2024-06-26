# DATASET
http://vrp.galgos.inf.puc-rio.br/index.php/en/

# CRVP-ACO

Este é um projeto de otimização de roteamento de veículos usando o algoritmo de otimização de colônia de formigas (ACO).

## Instalação

### 1. Clonar o Projeto

```bash
git clone https://github.com/gferreiracunha123/CRVP-ACO.git
cd CRVP-ACO
```

## Instalação anaconda windows
```bash
https://www.anaconda.com/download#downloads
```

### 1. Configurar o Ambiente
Recomenda-se criar um ambiente virtual para isolar as dependências do projeto.
```bash
conda create --name crvp-aco python
```
```bash
cd ~/Documents/CRVP-ACO
```
```bash
conda init
conda activate crvp-aco
```

### 2. Instalar Dependências
```bash
conda install --file requirements.txt
```
```bash
pip install --upgrade networkx
```

### Para executar o projeto, utilize o seguinte comando:
       --ite" default=20, help="Número de iterações")
       --ants" default=20, help="Número de formigas")
       --evapor", default=0.1, help="Taxa de evaporação")
      --k", default=3, help="Parâmetro k")
      --worst", help="Indicador para usar a pior formiga")
      --elitist", help="Indicador para usar a melhor formiga")
      --AG", help="Inicia a popualação inicial com Algoritimo genetico")
      --num_populations", help="Número de populações AG")
      --total_iters" default=100, help="Total de iterações AG")

```bash
python main.py --ite 20 --ants 20 --evapor 0.1 -k 5 --worst  --elitist --num_populations 1 --total_iters 100 --AG
```

### Para criar o plot dos arquivos de resultado basta executar:
```bash
python main_plot.py
```

### Contribuição
Se você quiser contribuir para o projeto, siga os passos abaixo:

Faça um fork do projeto.
Crie um branch para sua contribuição:
```bash
git checkout -b minha-contribuicao
git commit -m "Minha contribuição"
git push origin minha-contribuicao


