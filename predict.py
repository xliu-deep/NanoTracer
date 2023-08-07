import pandas as pd
import pickle
import csv
from xgboost import XGBClassifier
import networkx as nx
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from hiclass import LocalClassifierPerNode, LocalClassifierPerParentNode, LocalClassifierPerLevel
from hiclass import Classifier
from hiclass.Classifier import NodeClassifier
from sklearn.metrics import f1_score
from hiclass.metrics import f1,precision,recall
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns



def predict(hiclass):
	# load model parameters
	data = pd.read_csv(f'models/{hiclass}/BayesionOptimazation.log',
			header = 0, index_col = None)
	new_data =data.sort_values(by='F1',ascending=False)
	learning_rate,n_estimators,max_depth = new_data.iloc[0,:3]

	xgb = XGBClassifier(max_depth=int(max_depth), learning_rate=learning_rate,
							n_estimators=int(n_estimators), booster='gbtree')

	if hiclass == 'lcpl':
		model = LocalClassifierPerLevel(local_classifier=xgb)
		model.fit(X, y)  # placeholder must be explicited
	elif hiclass == 'lcppn':
		model = LocalClassifierPerParentNode(local_classifier=xgb)
		model.fit(X, y)
	elif hiclass == 'lcpn':
		model = LocalClassifierPerNode(local_classifier=xgb)
		model.fit(X, y)
	elif hiclass == 'flat':
		model = xgb
		model.fit(X, y_flat)

	y_pred = model.predict(X_test)	# test set
	print(y_pred)


	if hiclass == 'flat':
		y_true = y_test_flat
	else:
		y_true = y_test
		
	# model performance on test set
	f1_hiclass = f1(y_true=y_true, y_pred=y_pred)
	pre_hiclass = precision(y_true=y_true, y_pred=y_pred)
	rec_hiclass = recall(y_true=y_true, y_pred=y_pred)
	print(f1_hiclass, pre_hiclass, rec_hiclass)
	return (f1_hiclass, pre_hiclass, rec_hiclass)


# test set example
data_files = ['data/X_test.txt']
label_files = ['data/y_test_hiclass.csv']
label_flat_files = ['data/y_test_flat.txt']


# load traning set X,y
X = np.loadtxt('data/X_train.txt',delimiter='\t')
y = []
with open('data/y_train_hiclass.csv', 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		y.append(row)
y = np.array(y)
y_flat = np.loadtxt('data/y_train_flat.txt',dtype='str')


# load test set X,y
i = 0
X_test = np.loadtxt(data_files[i], delimiter='\t')
y_test = []
with open(label_files[i], 'r') as csvfile:
	csvreader = csv.reader(csvfile)
	for row in csvreader:
		y_test.append(row)
y_test = np.array(y_test)
y_test_flat = np.loadtxt(label_flat_files[i], dtype='str')


# model performance on test set
df = pd.DataFrame(index=['flat', 'lcpn', 'lcpl', 'lcppn'], columns=['F1', 'Precision', 'Recall'])
df.loc['flat', :] = predict('flat')
df.loc['lcppn', :] = predict('lcppn')
df.loc['lcpl', :] = predict('lcpl')
df.loc['lcpn', :] = predict('lcpn')
print(df)
