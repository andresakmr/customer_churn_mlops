# Customer Churn - Projeto de MLOps #

Este projeto utiliza Redes Neurais para prever a probabilidade de rotatividade de clientes, integrando ferramentas de MLOps para garantir o rastreamento, reprodutibilidade e versionamento do modelo.

**Responsável**: Andresa Araújo

 **Objetivo**: Desenvolver um modelo capaz de identificar clientes com alto risco de cancelamento, permitindo que a equipe de retenção tome medidas proativas.

## Tecnologias e Ferramentas
Linguagem: Python 3.10+

Deep Learning: TensorFlow / Keras

Rastreamento de Experimentos: MLflow

Infraestrutura MLOps: DagsHub

Ambiente: Conda / Anaconda

Controle de Versão: Git / GitHub

## Estrutura do Repositório

src/: Scripts de treinamento e predição.

data/: Dataset utilizado (Customer-Churn-Records.csv).

environment.yml: Configuração do ambiente Conda.

requirements.txt: Dependências do Python via Pip.

## Ficha Técnica do Modelo
Algoritmo: Rede Neural Sequencial (Multilayer Perceptron).

Arquitetura: * Camada de Entrada (16 neurônios).

2 Camadas Ocultas (10 neurônios cada, ativação ReLU).

Camada de Saída (1 neurônio, ativação Sigmoid).

Métrica Final (Acurácia): ~86% (Versão realista, sem Data Leakage).

Dataset: 10.000 registros com 18 variáveis originais.

## Experimentos e Artefatos
As métricas detalhadas e as versões do modelo podem ser consultadas no meu painel do [DagsHub](https://dagshub.com/andresakmr/customer_churn_mlops/models)
