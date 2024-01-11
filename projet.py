import math
from pandas_datareader import dat as pdr
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
pd.options.mode.chained_assigment = None


acao = "MGLU3.SA"

inicio = "2016-12-31"
final = "2023-12-31"

dados_acao = pdr.get_data_yahoo(acao, inicio, final)

dados_acao

cotacao = dados_acao['close'].to_numpy().reshape(-1,1)

cotacao

tamanho_dados_treinamento = int(len(cotacao)* 0.8)

tamanho_dados_treinamento
