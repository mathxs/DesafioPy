# -*- coding: utf-8 -*-

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

bd_train = pd.read_csv('/home/mathxs/Documentos/Projetos do Git/DesafioPy/testfiles/train.csv')
bd_train = pd.DataFrame(bd_train, columns=['Q047', 'Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'TP_SEXO','NU_NOTA_MT',
                                           'CO_UF_RESIDENCIA', 'NU_IDADE', 'TP_COR_RACA', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU',
                                           'TP_ESCOLA', 'TP_ENSINO', 'TP_PRESENCA_CH', 'TP_PRESENCA_LC', 'NU_NOTA_CN', 'NU_NOTA_CH',
                                           'NU_NOTA_LC', 'TP_LINGUA', 'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2',
                                           'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO'])
bd_train.corr().to_csv('dados.csv')