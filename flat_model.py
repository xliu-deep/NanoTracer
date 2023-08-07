import os
import sys
from functools import partial
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score,precision_score,recall_score
from pyGPGO.covfunc import matern32
from pyGPGO.acquisition import Acquisition
from pyGPGO.surrogates.GaussianProcess import GaussianProcess
from pyGPGO.GPGO import GPGO



np.random.seed(1)
xgb = XGBClassifier()

# load the training data
y = np.loadtxt('data/y_train_flat.txt',dtype='str')
X = np.loadtxt('data/X_train.txt',delimiter='\t')
# X_train = np.random.rand(105,2)

loo = LeaveOneOut()



def BP_Search(learning_rate,n_estimators,max_depth):
	prediction = []
	xgb = XGBClassifier(max_depth=int(max_depth), learning_rate=learning_rate,
					n_estimators=int(n_estimators), booster='gbtree')
	
	# data split
	for train_index, test_index in loo.split(X):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]
		xgb.fit(X_train, y_train)
		prediction.append(xgb.predict(X_test))
	y_pred = np.concatenate((prediction), axis=0)
	mask_index = y == '/'

	f1 = f1_score(y, y_pred, average = 'micro')
	pre = precision_score(y, y_pred, average='micro')
	rec = recall_score(y, y_pred, average='micro')  # write the best result
	# write the best result
	with open(log_file,'a') as f:
		f.write(','.join([str(round(learning_rate,4)), str(n_estimators), 
			str(max_depth)])+','+ str('%.3f'%f1)+','+ str('%.3f'%pre)+','+ str('%.3f'%rec) + '\n')

	return f1



# restore the hyperparameter
if not os.path.exists('flat'):
	os.mkdir('flat')
log_file = 'flat/BayesionOptimazation.log'
with open(log_file,'w') as f:
	f.write(','.join(['learning_rate', 'n_estimators',
		'max_depth','F1','Precision','Recall'])+'\n')

cov = matern32()
gp = GaussianProcess(cov)
acq = Acquisition(mode='UCB')
model_params = {
	'learning_rate':('cont', [0.0,1.0]),
	'n_estimators':('int', [10.0, 100.0]),
	'max_depth':('int', [5.0,20.0])}
gpgo = GPGO(gp, acq, BP_Search, model_params)
gpgo.run(max_iter=30)

