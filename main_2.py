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
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from os.path import join
from collections import Counter

from utils import euclidian, plot_plain_separator

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
dataset_name="Clusters Linear Without Superposition"

#%% Generate classes

print('Generating classes')

samples = 800
features = 2

class_1 = np.random.normal(2, 1.5, [samples, features])
class_2 = np.random.normal(8, 1.5, [samples, features])

x = np.concatenate([class_1, class_2]) 
y = np.concatenate(([0 for _ in range(samples)], [1 for _ in range(samples)]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=random_seed)
y_train_backup = y_train

plt.scatter(class_1[:,0], class_1[:,1])
plt.scatter(class_2[:,0], class_2[:,1])

#plt.scatter(class_1[:,0], class_1[:,1])
#plt.scatter(class_2[:,0], class_2[:,1])

#%% Perceptron

#TODO: Inserir Perceptron pra mostrar separação ideal

#%% Define model for classification

print('Building model')
# Do not interrupt the training before end of the epochs, to force an 
# overfitting

n_epochs=50000

model_1 = {
        'hidden_layer_sizes': (500,),
        'activation': 'relu',
        'alpha': 0,
        'max_iter': n_epochs,
        'random_state': random_seed,
        'tol':1e-4,
        'verbose': True,
        'n_iter_no_change': n_epochs
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
        'tol':1e-4,
        'verbose': True,
        'n_iter_no_change': n_epochs
        }

model_4 = {
        'hidden_layer_sizes': (100,),
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

#%%
#ss = ShuffleSplit(n_splits=5, test_size=0.3, random_state=random_seed)
#cv_results = cross_validate(model, x, y, cv=ss, n_jobs = 6, verbose=2, return_train_score=True)
#print(cv_results)

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

#%% Plot separator

print('Saving plot for model without adjustment')

plt.clf()

class_1_new = x_train[y_train==0]
class_2_new = x_train[y_train==1]

plt.scatter(class_1_new[:,0], class_1_new[:,1])
plt.scatter(class_2_new[:,0], class_2_new[:,1])
plot_plain_separator(model, x, save=fr'{"_".join(dataset_name.lower().split())}_nao_ajustada')

#%% Get distance matrix

#distance_matrix = np.zeros((x.shape[0], x.shape[0]))
#
#for sample_a in range(x.shape[0]):
#    
#    if sample_a > 0 and sample_a % int(0.1 * x.shape[0]) == 0:
#        print(f'Progress: {sample_a / int(0.1 * x.shape[0]) * 10} % '
#              f'({sample_a} samples from {x.shape[0]} total samples)')
#    
#    for sample_b in range(x.shape[0]):
#        distance_matrix[sample_a, sample_b] = euclidian(x[sample_a], x[sample_b])
#
#classes = list(Counter(y).keys())

#%% Use KNN to find unsure samples

print('Performing ajustment with KNN')

classifier = KNeighborsClassifier(n_neighbors=10)

y_classes = []
for index in range(x_train.shape[0]):
    x_t = np.delete(x_train, index, 0)
    y_t = np.delete(y_train, index, 0)
    classifier.fit(x_t, y_t)
    y_classes.append(classifier.predict([x_train[index]])[0])

errors = y_train - y_classes

wrong_classes = np.where(errors != 0)[0]

print('Fixing training samples with adjustment')

y_train = y_classes

#for i in wrong_classes:
#    y_train[i] = 0 if y_train[i] == 1 else 1

#%%
#x_h = np.concatenate([x_train, x_test])
#y_h = np.concatenate([y_train, y_test])
#cv_results_2 = cross_validate(model, x_h, y_h, cv=3, n_jobs = 4, verbose=4, return_train_score=True)
#print(cv_results)

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

#%% Plot separator

print('Saving figure for model adjusted')

plt.clf()
class_1_new = x_train[y_train==0]
class_2_new = x_train[y_train==1]

plt.scatter(class_1_new[:,0], class_1_new[:,1])
plt.scatter(class_2_new[:,0], class_2_new[:,1])
plot_plain_separator(model, x, save=fr'{"_".join(dataset_name.lower().split())}_ajustada')

