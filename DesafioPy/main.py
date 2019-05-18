# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

bd_train = pd.read_csv('/home/mathxs/Documentos/Projetos do Git/DesafioPy/testfiles/train.csv')
#bd_train.columns
#bd_train = bd_train.dropna(axis=1, how='all');
#bd_train = bd_train.dropna(axis=0, subset=['NU_NOTA_MT', 'CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'IN_TREINEIRO', 'TP_DEPENDENCIA_ADM_ESC', 'IN_BAIXA_VISAO', 'IN_SURDEZ', 'IN_DISLEXIA', 'IN_DISCALCULIA', 'IN_SABATISTA', 'IN_GESTANTE', 'IN_IDOSO', 'TP_PRESENCA_CN', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO'])
bd_train = pd.DataFrame(bd_train, columns=['Q047', 'Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'TP_SEXO','NU_NOTA_MT', 'CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO'])
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

bd_train = bd_train.replace('c8328ebc6f3238e06076c481bc1b82b8301e7a3f',100)
bd_train = bd_train.replace('b9b06ce8c319a3df2158ea3d0aef0f7d3eecaed7',101)
bd_train = bd_train.replace('2d22ac1d42e6187f09ee6c578df187a760123ccf',102)
bd_train = bd_train.replace('c8328ebc6f3238e06076c481bc1b82b8301e7a3f',103)
bd_train = bd_train.replace('66b1dad288e13be0992bae01e81f71eca1c6e8a6',104)
bd_train = bd_train.replace('03b1fba5c1ebbc47988cd303b08982cfb2aa9cf2',105)
bd_train = bd_train.replace('c87a85497686b3e7b3765f84a2ca95256f0f66aa',106)
bd_train = bd_train.replace('69ed2ddcb151cfebe3d2ae372055335ac7c8c144',107)
bd_train = bd_train.replace('1bcdece8fb1b952552b319e4e5512bbcf540e338',108)
bd_train = bd_train.replace('a27a1efea095c8a973496f0b57a24ac6775d95b0',109)
bd_train = bd_train.replace('9cd70f1b922e02bd33453b3f607f5a644fb9b1b8',110)
bd_train = bd_train.replace('909237ab0d84688e10c0470e2997348aff585273',111)
bd_train = bd_train.replace('f48d390ab6a2428e659c37fb8a9d00afde621889',112)
bd_train = bd_train.replace('942ab3dc020af4cf53740b6b07e9dd7060b24164',113)
bd_train = bd_train.replace('f94e97c2a5689edd5369740fde9a927e23a9465f',114)
bd_train = bd_train.replace('0fb4772fc6ee9b951ade2fbe6699cc37985c422e',115)
bd_train = bd_train.replace('c95541bf218d7ff70572ca4bcb421edeff05c6d5',116)
bd_train = bd_train.replace('6c3fec2ef505409a9e7c3d2e8634fa2aced4ee93',117)
bd_train = bd_train.replace('d5f6d17523d2cce3e4dc0a7f0582a85cec1c15ee',118)
bd_train = bd_train.replace('01af53cd161a420fff1767129c10de560cc264dd',119)
bd_train = bd_train.replace('01abbb7f1a90505385f44eec9905f82ca2a42cfd',120)
bd_train = bd_train.replace('5aebe5cad7fabc1545ac7fba07a4e6177f98483c',121)
bd_train = bd_train.replace('72f80e4b3150c627c7ffc93cfe0fa13a9989b610',122)
bd_train = bd_train.replace('9cbf6bf31d9d89a64ce2737ece4834fde4a95029',123)
bd_train = bd_train.replace('fa86b01f07636b15adfd66b688c79934730721a6',124)
bd_train = bd_train.replace('44b09b311799bd684b3d02463bfa99e472c6adb3',125)
bd_train = bd_train.replace('481058938110a64a272266e3892102b8ef0ca96f',126)
bd_train = bd_train.replace('97caab1e1533dba217deb7ef41490f52e459ab01',127)
bd_train = bd_train.replace('81d0ee00ef42a7c23eb04496458c03d4c5b9c31a',128)
bd_train = bd_train.replace('767a32545304ed293242d528f54d4edb1369f910',129)
bd_train = bd_train.replace('577f8968d95046f5eb5cc158608e12fa9ba34c85',130)
bd_train = bd_train.replace('0ec1c8ac02d2747b6e9a99933fbf96127dd6e89e',131)
bd_train = bd_train.replace('0e0082361eaceb6418bb17305a2b7912650b4783',132)
bd_train = bd_train.replace('6d6961694e839531aec2d35bbd8552b55394a0d7',133)
bd_train = bd_train.replace('73c5c86eef8f70263e4c5708d153cca123f93378',134)
bd_train = bd_train.replace('16f84b7b3d2aeaff7d2f01297e6b3d0e25c77bb2',135)


bd_train = bd_train.fillna('-10')
bd_train1 = bd_train[bd_train['TP_PRESENCA_CH'] != 2]
bd_train2 = bd_train1[bd_train1['TP_PRESENCA_LC'] == 1] 
 
#sns.distplot(bd_train['NU_NOTA_MT'], bd_train['NU_NOTA_CH'], bd_train['NU_NOTA_LC'], bd_train['NU_NOTA_CN']);
#sns.distplot(bd_train['NU_NOTA_MT']);
columns=['Q047','CO_PROVA_CH', 'CO_PROVA_MT', 'Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'TP_SEXO','NU_NOTA_MT', 'CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']
'''
for col in columns:
    #plt.hist(bd_train[col], normed=True, alpha=0.5)
    sns.kdeplot(bd_train[col], shade=True)
'''
#Regressão linear
# Preparando os Dados
#X = bd_train[['NU_NOTA_CH']]
X = bd_train2[['Q047', 'Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'TP_SEXO','CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']]
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

bd_erro = pd.DataFrame.from_records(y_test)
bd_erro = bd_erro.join(pd.DataFrame.from_records(y_pred))


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
bd_result = pd.DataFrame(bd_result, columns=['Q047', 'Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'TP_SEXO','NU_INSCRICAO', 'CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO' , 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO'])
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

bd_result = bd_result.replace('01abbb7f1a90505385f44eec9905f82ca2a42cfd',100)
bd_result = bd_result.replace('01af53cd161a420fff1767129c10de560cc264dd',101)
bd_result = bd_result.replace('03b1fba5c1ebbc47988cd303b08982cfb2aa9cf2',102)
bd_result = bd_result.replace('0e0082361eaceb6418bb17305a2b7912650b4783',103)
bd_result = bd_result.replace('0ec1c8ac02d2747b6e9a99933fbf96127dd6e89e',104)
bd_result = bd_result.replace('0fb4772fc6ee9b951ade2fbe6699cc37985c422e',105)
bd_result = bd_result.replace('16f84b7b3d2aeaff7d2f01297e6b3d0e25c77bb2',106)
bd_result = bd_result.replace('1bcdece8fb1b952552b319e4e5512bbcf540e338',107)
bd_result = bd_result.replace('2d22ac1d42e6187f09ee6c578df187a760123ccf',108)
bd_result = bd_result.replace('44b09b311799bd684b3d02463bfa99e472c6adb3',109)
bd_result = bd_result.replace('481058938110a64a272266e3892102b8ef0ca96f',110)
bd_result = bd_result.replace('577f8968d95046f5eb5cc158608e12fa9ba34c85',111)
bd_result = bd_result.replace('5aebe5cad7fabc1545ac7fba07a4e6177f98483c',112)
bd_result = bd_result.replace('66b1dad288e13be0992bae01e81f71eca1c6e8a6',113)
bd_result = bd_result.replace('69ed2ddcb151cfebe3d2ae372055335ac7c8c144',114)
bd_result = bd_result.replace('6c3fec2ef505409a9e7c3d2e8634fa2aced4ee93',115)
bd_result = bd_result.replace('6d6961694e839531aec2d35bbd8552b55394a0d7',116)
bd_result = bd_result.replace('72f80e4b3150c627c7ffc93cfe0fa13a9989b610',117)
bd_result = bd_result.replace('73c5c86eef8f70263e4c5708d153cca123f93378',118)
bd_result = bd_result.replace('767a32545304ed293242d528f54d4edb1369f910',119)
bd_result = bd_result.replace('81d0ee00ef42a7c23eb04496458c03d4c5b9c31a',120)
bd_result = bd_result.replace('909237ab0d84688e10c0470e2997348aff585273',121)
bd_result = bd_result.replace('942ab3dc020af4cf53740b6b07e9dd7060b24164',122)
bd_result = bd_result.replace('97caab1e1533dba217deb7ef41490f52e459ab01',123)
bd_result = bd_result.replace('9cbf6bf31d9d89a64ce2737ece4834fde4a95029',124)
bd_result = bd_result.replace('9cd70f1b922e02bd33453b3f607f5a644fb9b1b8',125)
bd_result = bd_result.replace('b9b06ce8c319a3df2158ea3d0aef0f7d3eecaed7',126)
bd_result = bd_result.replace('c8328ebc6f3238e06076c481bc1b82b8301e7a3f',127)
bd_result = bd_result.replace('c87a85497686b3e7b3765f84a2ca95256f0f66aa',128)
bd_result = bd_result.replace('c95541bf218d7ff70572ca4bcb421edeff05c6d5',129)
bd_result = bd_result.replace('f48d390ab6a2428e659c37fb8a9d00afde621889',130)
bd_result = bd_result.replace('f94e97c2a5689edd5369740fde9a927e23a9465f',131)
bd_result = bd_result.replace('fa86b01f07636b15adfd66b688c79934730721a6',132)


bd_result_join = bd_result[bd_result['TP_PRESENCA_LC'] == 1] 


X_result = bd_result[['Q047', 'Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'TP_SEXO','CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO']]
bd_result_join = bd_result_join.drop(columns=['Q047', 'TP_PRESENCA_LC','Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'TP_SEXO','CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'TP_PRESENCA_CH', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO'])

y_result = regressor.predict(X_result)
bd_y = pd.DataFrame.from_records(y_result)
bd_y.columns=['NU_NOTA_MT']
bd_y = bd_y.NU_NOTA_MT.round(1)

bd_result_join = bd_result_join.join(bd_y)

bd_result_join.to_csv('answer.csv',index=False)
