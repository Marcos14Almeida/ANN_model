# ANN_model

## Descrição do Projeto 

Modelo de rede neural ANN para detectar se um usuário permanece ou não no banco. Esse algoritmo faz parte de um curso que fiz, mas adaptei e melhorei algumas etapas.

## Como usar o Projeto 

Através do uso de Python, eu uso um dataset com informações a respeito de clientes de um banco. A partir disso, eu filtro o dataset em train_set e test_set aplico feature scaling e o método de rede neural ANN no train_set com a biblioteca Keras para obter uma previsão dos resultados. Modificando os hiperparâmetros e observando as métricas de Confusion Matrix e Accuracy, o algoritmo foi aperfeiçoado para melhores resultados.

## Como Executar o projeto

Baixe os arquivos e com o uso de algum compilador python com suporte para Keras, Tensorflow, Numpy, Pandas e Sklearn instalados execute o código do arquivo ".py".

Para instalar o Tensorflow no compilador python tente esse link: https://medium.com/@pushkarmandot/installing-tensorflow-theano-and-keras-in-spyder-84de7eb0f0df

## Resultado

Confusion Matrix = 
[[46  7]
 [19  6]]
Accuracy = 0.666
