# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 19:22:55 2019

@author: victor
"""

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import os
from os.path import join
from collections import Counter
import csv
import click

from utils import euclidian, plot_plain_separator

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
dataset_name="Adult Income Cleaned 2"

#%% Generate classes

print('Generating classes')

df = pd.DataFrame
with open(join('custom_datasets', 'adult.data')) as file:
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
             'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hpw', 'native_country',
             'class']
    df = pd.read_csv(file, names=names)

df = df.apply(lambda r: [None if item == ' ?' else item for item in r])
df.dropna(inplace=True)

def convert(column):
    result = pd.Series()
    try:
        float()
        result = column.astype(np.float32)
    except:
        le = LabelEncoder()
        result = le.fit_transform(column)
    
    return result

df = df.apply(convert)

x = df.iloc[:, 0:-1].values.astype(np.float32)
y = df['class'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_seed)
y_train_backup = y_train


#%% Define model for classification

print('Building model')
# Do not interrupt the training before end of the epochs, to force an 
# overfitting

n_epochs=10000

model_1 = {
        'hidden_layer_sizes': (500,),
        'activation': 'relu',
        'max_iter': n_epochs,
        'random_state': random_seed,
        'tol':1e-4,
        'verbose': True,
        'batch_size': 600
        }

model_2 = {
        'hidden_layer_sizes': (500,),
        'activation': 'relu',
        'alpha': 0,
        'shuffle': False,
        'batch_size': 1,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.05,
        'max_iter': n_epochs,
        'random_state': random_seed,
        'tol':1e-4,
        'verbose': True,
        'n_iter_no_change': n_epochs
        }

model_3 = {
        'hidden_layer_sizes': (500,),
        'solver': 'sgd',
        'activation': 'tanh',
        'alpha': 0,
        'learning_rate': 'adaptive',
        'learning_rate_init': 0.05,
        'max_iter': n_epochs,
        'random_state': random_seed,
        'batch_size': 500,
        'tol':1e-4,
        'verbose': True,
        'n_iter_no_change': n_epochs
        }

model_4 = {
        'hidden_layer_sizes': (50,),
        'activation': 'relu',
        'alpha': 0,
        'max_iter': n_epochs,
        'random_state': random_seed,
        'tol':1e-12,
        'verbose': True,
        'n_iter_no_change': n_epochs
        }

model = MLPClassifier(**model_4)

#%% Train model

print('Training model without adjustment')
model.fit(x_train, y_train)

#%% Test

print('Getting score for model without adjustment')

score_test = model.score(x_test, y_test)
score_train = model.score(x_train, y_train)
norm_weights_hidden = np.linalg.norm(model.coefs_[0])
norm_weights_out = np.linalg.norm(model.coefs_[1])

print(f'[RESULT]\nScore Train = {score_train:0.4f}'
      f'\nScore test = {score_test:0.4f}'
      f'\nRatio = {score_test/score_train:0.4f}'
      f'\nNorm Weights Hidden = {norm_weights_hidden}'
      f'\nNormWeights Out = {norm_weights_out}')

y_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%% Save result tables as Latex format

print('Saving latex tables')

cm = confusion_matrix(y_test, y_pred)
test_name=f'Treinamento sem Ajuste {dataset_name}'
table_confusion = r"""
\begin{{table}}[H]
\centering
\caption{{Matrix de confusão com dataset de treinamento para {test_name} }}
\label{{tab:matrix confusao {test_name} }}
\begin{{tabular}}{{cc}}
\hline
\multicolumn{{2}}{{c}}{{ Matriz de Confusão para {test_name} }} \\ \hline
{tp} & {fp} \\
{fn} & {tn} \\ \hline
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Acurácia para {test_name} }}
\label{{tab:acuracia {test_name} }}
\begin{{tabular}}{{lr}}
\hline
\multicolumn{{2}}{{c}}{{Acurácia}} \\ \hline
Conjunto de Treinamento & {score_train:0.4f} \\
Conjunto de teste & {score_test:0.4f} \\
Razão & {score_ratio:0.4f} \\ \hline
\multicolumn{{2}}{{c}}{{Norma dos Pesos}} \\ \hline
Camada escondida & {norm_weights_hidden} \\
Camada de saída & {norm_weights_out} \\ \hline
\end{{tabular}}
\end{{table}}

""".format(
    test_name=test_name,
    tp=cm[0,0],
    fp=cm[0,1],
    fn=cm[1,0],
    tn=cm[1,1],
    score_train=score_train,
    score_test=score_test,
    score_ratio=score_test/score_train,
    norm_weights_hidden=norm_weights_hidden,
    norm_weights_out=norm_weights_out
)

with open(join('..', 'Artigo_1_RNA', 'tables', fr'{"_".join(test_name.lower().split())}.txt'), 'w') as file:
    file.write(table_confusion)

#%% Use KNN to find unsure samples

print('Performing ajustment with KNN')

classifier = KNeighborsClassifier(n_neighbors=20)

y_classes = []
with click.progressbar(range(x_train.shape[0])) as indexes:
    for index in indexes:
        x_t = np.delete(x_train, index, 0)
        y_t = np.delete(y_train, index, 0)
        classifier.fit(x_t, y_t)
        y_classes.append(classifier.predict([x_train[index]])[0])

errors = y_train - y_classes

wrong_classes = np.where(errors != 0)[0]

print('Fixing training samples with adjustment')

y_train = y_classes


#%% Retrain
    
print('Retraining model after adjustment')

model.fit(x_train, y_train)

score_test = model.score(x_test, y_test)
score_train = model.score(x_train, y_train)
norm_weights_hidden = np.linalg.norm(model.coefs_[0])
norm_weights_out = np.linalg.norm(model.coefs_[1])

print(f'[RESULT]\nScore Train = {score_train:0.4f}'
      f'\nScore test = {score_test:0.4f}'
      f'\nRatio = {score_test/score_train:0.4f}'
      f'\nNorm Weights Hidden = {norm_weights_hidden}'
      f'\nNormWeights Out = {norm_weights_out}')

y_pred = model.predict(x_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%% Save result tables as Latex format

print('Saving latex tables for model adjusted')

cm = confusion_matrix(y_test, y_pred)
test_name=f'Treinamento com Ajuste {dataset_name}'
table_confusion = r"""
\begin{{table}}[H]
\centering
\caption{{Matrix de confusão com dataset de treinamento para {test_name} }}
\label{{tab:matrix confusao {test_name} }}
\begin{{tabular}}{{cc}}
\hline
\multicolumn{{2}}{{c}}{{ Matriz de Confusão para {test_name} }} \\ \hline
{tp} & {fp} \\
{fn} & {tn} \\ \hline
\end{{tabular}}
\end{{table}}

\begin{{table}}[H]
\centering
\caption{{Acurácia para {test_name} }}
\label{{tab:acuracia {test_name} }}
\begin{{tabular}}{{lr}}
\hline
\multicolumn{{2}}{{c}}{{Acurácia}} \\ \hline
Conjunto de Treinamento & {score_train:0.4f} \\
Conjunto de teste & {score_test:0.4f} \\
Razão & {score_ratio:0.4f} \\ \hline
\multicolumn{{2}}{{c}}{{Norma dos Pesos}} \\ \hline
Camada escondida & {norm_weights_hidden} \\
Camada de saída & {norm_weights_out} \\ \hline
\end{{tabular}}
\end{{table}}

""".format(
    test_name=test_name,
    tp=cm[0,0],
    fp=cm[0,1],
    fn=cm[1,0],
    tn=cm[1,1],
    score_train=score_train,
    score_test=score_test,
    score_ratio=score_test/score_train,
    norm_weights_hidden=norm_weights_hidden,
    norm_weights_out=norm_weights_out
)

with open(join('..', 'Artigo_1_RNA', 'tables', fr'{"_".join(test_name.lower().split())}.txt'), 'w') as file:
    file.write(table_confusion)
