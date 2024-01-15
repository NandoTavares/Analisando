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

#agora é tranformar as listas em arrayas e dando reshape 3d
        
treinamento_x, treinamento_y = np.array(treinamento_x), np.array(treinamento_y)
treinamento_x

treinamento_x = treinamento_x.reshape(treinamento_x.shape[0], treinamento_x.shape[1], 1)
treinamento_x

modelo = Sequential()

modelo.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
modelo.add(LSTM(50, return_sequences=False))
modelo.add(Dense(25))
modelo.add(Dense(1))

modelo.compile(optimizer='adam', loss='mean_squared_error')

modelo.fit(treinamento_x, treinamento_y, epochs=1, batch_size=1)

dados_teste = dados_entre_0_e_1[tamanho_dados_treinamento - 60:, :]

teste_x = []
teste_y = cotacao[tamanho_dados_treinamento: , :]

for i in range(60, len(dados_teste)):
    teste_x.append(dados_teste[i - 60: i, 0])

#agora o reshape

teste_x = np.array(teste_x)
teste_x = teste_x.reshape(teste_x.shape[0], teste_x.shape[1], 1)

predicoes = modelo.predict(teste_x) 
predicoes = escalador.inverse_transform(predicoes)
predicoes #pegando as prefições e tirando a escala dos dados

rmse = np.sqrt(np.mean(predicoes - teste_y) **2)
rmse #pegando o erro medio quadratico

treinamento = dados_acao.iloc[:tamanho_dados_treinamento, :]
df_teste = pd.DataFrame({"Close": dados_acao['Close'].iloc[tamanho_dados_treinamento:],
                         "predicoes":predicoes.reshape(len(predicoes))})

plt.figure(figsize = (16,8))
plt.title('Modelo')
plt.xlabel('Data', fontsize = 18)
plt.ylabel("Preço de fechamento", fontsize = 18)
plt.plot(treinamento['Close'])
plt.plot(df_teste[['Close', 'predicoes']])
plt.legend(['Treinamento', 'Real', 'Predições'], loc=2, prop={'size':16})
plt.show()

df_teste.sort_index()
df_teste

df_teste['variacao_percentual_acao'] = df_teste['Close'].pct_change()
df_teste['variacao_percentual_modelo'] = df_teste["predicoes"].pct_change()

df_teste = df_teste.dropna()

df_teste['var_acao_maior_menor_que_zero'] = np.where(df_teste['variacao_percentual_acao'] > 0,
                                                     True, False)
df_teste['var_modelo_maior_menor_que_zero'] = np.where(df_teste['variacao_percentual_modelo'] > 0,
                                                     True, False)
df_teste['acertou_o_lado'] = np.where(df_teste['var_acao_maior_menor_que_zero'] == df_teste['var_modelo_maior_menor_que_zero'],
                                      True, False)
df_teste['variacao_percentual_acao_abs'] = df_teste['variacao_percentual_acao'].abs()
df_teste

acertou_o_lado = df_teste['acertou_o_lado'].sum()/len(df_teste['acertou_o_lado'])
errou_lado = 1 - acertou_o_lado

media_lucro = df_teste.groupby('acertou_o_lado')['variacao_percentual_acao_abs'].mean()
exp_mat_lucro = acertou_o_lado * media_lucro.iloc[1] - media_lucro.iloc[0] * errou_lado

ganho_sobre_perda = media_lucro.iloc[1] / media_lucro.iloc[0]

print(media_lucro,ganho_sobre_perda, acertou_o_lado, (exp_mat_lucro * 100))

data_hoje = datetime.now()

#se quiser escolher um dia tu faz tipo
#data_hoje = datetime.now() - timedelta(days = 1)

if data_hoje.hour > 18:
    final = data_hoje
    inicial = datetime.now() - timedelta(days = 252)

else:
    final = data_hoje - timedelta(days = 1)
    inicial = datetime.now() - timedelta(days = 252)


cotacoes = yf.download(acao, start=inicial, end=final)
ultimos_60_dias = cotacao[-60:].reshape(-1, 1)
ultimos_60_dias_escalado = escalador.transform(ultimos_60_dias)

teste_x = []
teste_x.append(ultimos_60_dias_escalado)
teste_x = np.array(teste_x)
teste_x = teste_x.reshape(teste_x.shape[0], teste_x.shape[1], 1)

previsao_de_preco = modelo.predict(teste_x)
previsao_de_preco = escalador.inverse_transform(previsao_de_preco)

print(previsao_de_preco)
