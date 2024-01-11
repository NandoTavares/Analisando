import math
from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
pd.options.mode.chained_assignment = None


acao = "MGLU3.SA"

inicio = "2016-12-31"
final = "2023-12-31"

dados_acao = yf.download(acao, start=inicio, end=final)

dados_acao

cotacao = dados_acao['Close'].to_numpy().reshape(-1, 1)

cotacao

tamanho_dados_treinamento = int(len(cotacao)* 0.8)

tamanho_dados_treinamento

escalador = MinMaxScaler(feature_range=(0, 1))

dados_entre_0_e_1_treinamento = escalador.fit_transform(cotacao[0: tamanho_dados_treinamento, :])
dados_entre_0_e_1_teste = escalador.transform(cotacao[tamanho_dados_treinamento: , :])
dados_entre_0_e_1 = list(dados_entre_0_e_1_treinamento.reshape(
    len(dados_entre_0_e_1_treinamento))) + list(dados_entre_0_e_1_teste.reshape(len(dados_entre_0_e_1_teste)))

dados_entre_0_e_1 = np.array(dados_entre_0_e_1).reshape(len(dados_entre_0_e_1), 1)
dados_entre_0_e_1

dados_para_treinamento = dados_entre_0_e_1[0: tamanho_dados_treinamento, :]
treinamento_x = []
treinamento_y = []

for i in range(60,len(dados_para_treinamento)):
    treinamento_x.append(dados_para_treinamento[i - 60:i, 0]) #ulimos 60 dias
    treinamento_y.append(dados_para_treinamento[i, 0])

    if i <= 61:
        print(treinamento_x)
        print(treinamento_y)
