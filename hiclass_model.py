import os
import sys
import csv
from functools import partial

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import networkx as nx

from hiclass.metrics import f1,precision,recall
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from hiclass.metrics import f1,precision,recall
from sklearn.model_selection import LeaveOneOut

from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO



np.random.seed()
xgb = XGBClassifier()


# load the training data
X = np.loadtxt('data/X_train.txt',delimiter='\t')
y = []
with open('data/y_train_hiclass.csv', 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		y.append(row)
y = np.array(y)


loo = LeaveOneOut()


def BP_Search(mcc_model, learning_rate,n_estimators,max_depth):
	prediction = []
	# print(learning_rate,n_estimators,max_depth)
	xgb = XGBClassifier(max_depth=int(max_depth), learning_rate=learning_rate,
					n_estimators=int(n_estimators), booster='gbtree')
	
	# data split
	for train_index, test_index in loo.split(X):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		model = getattr(sys.modules[__name__], mcc_model)(local_classifier=xgb)
		model.fit(X_train, y_train)
		prediction.append(model.predict(X_test))
	y_pred = np.concatenate((prediction), axis=0)
	f1_score = f1(y_true =y, y_pred = y_pred)
	
	# write the best result
	with open(log_file,'a') as f:
		f.write(','.join([str(round(learning_rate,4)), str(n_estimators), 
			str(max_depth)])+','+ str('%.3f'%f1_score) + '\n')
	return f1_score


lcpl = ['lcpl', 'LocalClassifierPerLevel']
lcppn = ['lcppn', 'LocalClassifierPerParentNode']
lcpn = ['lcpn', 'LocalClassifierPerNode']
hiclass_models = [lcpl,lcppn,lcpn]


for hiclassmodel in hiclass_models:

	# restore the hyperparameter
	if not os.path.exists(f'models/{hiclassmodel[0]}'):
		os.mkdir(f'models/{hiclassmodel[0]}')
	log_file = f'models/{hiclassmodel[0]}/BayesionOptimazation.log'
	with open(log_file,'w') as f:
		f.write(','.join(['learning_rate', 'n_estimators',
			'max_depth','F1'])+'\n')

	cov = matern32()
	gp = GaussianProcess(cov)
	acq = Acquisition(mode='UCB')
	model_params = {
		'learning_rate':('cont', [0.0,1.0]),
		'n_estimators':('int', [10.0, 100.0]),
		'max_depth':('int', [5.0,20.0])}
	gpgo = GPGO(gp, acq, partial(BP_Search, hiclassmodel[1]), model_params)
	gpgo.run(max_iter=50)

