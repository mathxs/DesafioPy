# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

bd_train = pd.read_csv('/home/mathxs/Documentos/Projetos do Git/DesafioPy/testfiles/train.csv')
#bd_train.columns
#bd_train = bd_train.dropna(axis=1, how='all');
#bd_train = bd_train.dropna(axis=0, subset=['NU_NOTA_MT', 'CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'IN_TREINEIRO', 'TP_DEPENDENCIA_ADM_ESC', 'IN_BAIXA_VISAO', 'IN_SURDEZ', 'IN_DISLEXIA', 'IN_DISCALCULIA', 'IN_SABATISTA', 'IN_GESTANTE', 'IN_IDOSO', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO'])
bd_train = pd.DataFrame(bd_train, columns=['Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'Q047', 'IN_TREINEIRO', 'TP_NACIONALIDADE', 'TP_SEXO','NU_NOTA_MT', 'CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC', 'IN_DISCALCULIA', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO'])
bd_train = bd_train.replace('M','0')
bd_train = bd_train.replace('F','1')
bd_train = bd_train.replace('A','0')
bd_train = bd_train.replace('B','1')
bd_train = bd_train.replace('C','2')
bd_train = bd_train.replace('D','3')
bd_train = bd_train.replace('E','4')
bd_train = bd_train.replace('F','5')
bd_train = bd_train.replace('G','6')
bd_train = bd_train.replace('H','7')
bd_train = bd_train.replace('I','8')
bd_train = bd_train.replace('J','9')
bd_train = bd_train.replace('K','10')
bd_train = bd_train.replace('L','11')
bd_train = bd_train.replace('M','12')
bd_train = bd_train.replace('N','13')
bd_train = bd_train.replace('O','14')
bd_train = bd_train.replace('P','15')
bd_train = bd_train.replace('Q','16')

bd_train = bd_train.fillna('-10')
bd_train1 = bd_train[bd_train['TP_PRESENCA_CN'] != 2]
bd_train2 = bd_train1[bd_train1['TP_PRESENCA_LC'] == 1] 
 
#sns.distplot(bd_train['NU_NOTA_MT'], bd_train['NU_NOTA_CH'], bd_train['NU_NOTA_LC'], bd_train['NU_NOTA_CN']);
#sns.distplot(bd_train['NU_NOTA_MT']);
columns=['Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'IN_TREINEIRO', 'TP_NACIONALIDADE', 'TP_SEXO','NU_NOTA_MT', 'CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC', 'IN_DISCALCULIA', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']
'''
for col in columns:
    #plt.hist(bd_train[col], normed=True, alpha=0.5)
    sns.kdeplot(bd_train[col], shade=True)
'''
#Regressão linear
# Preparando os Dados
#X = bd_train[['NU_NOTA_CH']]
X = bd_train2[['Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'IN_TREINEIRO', 'TP_NACIONALIDADE', 'TP_SEXO','CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC', 'IN_DISCALCULIA', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']]
y = bd_train2[['NU_NOTA_MT']]
	
# Separando o conjunto de dados em Conjunto de Treino e Validação
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

#Treinando o Modelo de Regressão Linear
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
# Ajustando os Dados de Treinamento ao nosso Modelo
lr = regressor.fit(X, y)
# Predizendo o preço para nosso Conjunto de Validação
y_pred = regressor.predict(X_test)
# Pontuando o Modelo

from sklearn.metrics import r2_score, mean_squared_error
# Valor de R2 perto de 1 nos diz que é um bom modelo
print(f"R2 score: {r2_score(y_test, y_pred)}")
# MSE Score perto de 0 é um bom modelo
print(f"MSE score: {mean_squared_error(y_test, y_pred)}")

'''
#plotando o grafico
linha = np.linspace(-3, 3, 500).reshape(-1, 1) #Para gerar a linha da predição
plt.plot(X_train, y_train, "^", markersize = 5)
plt.plot(X_test, y_test, "v", markersize = 7)
plt.plot(linha, lr.predict(linha))
plt.grid(True)
'''

bd_result = pd.read_csv('/home/mathxs/Documentos/Projetos do Git/DesafioPy/testfiles/test.csv')
#bd_train.columns
#bd_result = bd_result.dropna(axis=1, how='all');
#bd_result = bd_train.dropna(axis=0, subset=['NU_NOTA_MT'])
#bd_result = bd_result.dropna(axis=0, subset=['NU_INSCRICAO', 'CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'IN_TREINEIRO', 'TP_DEPENDENCIA_ADM_ESC', 'IN_BAIXA_VISAO', 'IN_SURDEZ', 'IN_DISLEXIA', 'IN_DISCALCULIA', 'IN_SABATISTA', 'IN_GESTANTE', 'IN_IDOSO', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO'])
bd_result = pd.DataFrame(bd_result, columns=['Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'IN_TREINEIRO', 'TP_NACIONALIDADE', 'TP_SEXO','NU_INSCRICAO', 'CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO' , 'TP_DEPENDENCIA_ADM_ESC', 'IN_DISCALCULIA', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO'])
bd_result = bd_result.fillna('-10')
bd_result = bd_result.replace('M','0')
bd_result = bd_result.replace('F','1')
bd_result = bd_result.replace('A','0')
bd_result = bd_result.replace('B','1')
bd_result = bd_result.replace('C','2')
bd_result = bd_result.replace('D','3')
bd_result = bd_result.replace('E','4')
bd_result = bd_result.replace('F','5')
bd_result = bd_result.replace('G','6')
bd_result = bd_result.replace('H','7')
bd_result = bd_result.replace('I','8')
bd_result = bd_result.replace('J','9')
bd_result = bd_result.replace('K','10')
bd_result = bd_result.replace('L','11')
bd_result = bd_result.replace('M','12')
bd_result = bd_result.replace('N','13')
bd_result = bd_result.replace('O','14')
bd_result = bd_result.replace('P','15')
bd_result = bd_result.replace('Q','16')
bd_result_join = bd_result[bd_result['TP_PRESENCA_LC'] == 1] 


X_result = bd_result[['Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026','IN_TREINEIRO', 'TP_NACIONALIDADE', 'TP_SEXO','CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC', 'IN_DISCALCULIA', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']]
bd_result_join = bd_result_join.drop(columns=['TP_PRESENCA_LC','Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'IN_TREINEIRO', 'TP_NACIONALIDADE', 'TP_SEXO','CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_DEPENDENCIA_ADM_ESC', 'IN_DISCALCULIA', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO'])

y_result = regressor.predict(X_result)
bd_y = pd.DataFrame.from_records(y_result)
bd_y.columns=['NU_NOTA_MT']
bd_y = bd_y.NU_NOTA_MT.round(1)

bd_result_join = bd_result_join.join(bd_y)

bd_result_join.to_csv('answer.csv',index=False)
