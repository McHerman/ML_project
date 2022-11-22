# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 14:10:37 2022

@author: thues
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV


pulsar = pd.read_csv('pulsar_data.csv')

features = ['IP mean', 'IP stdev', 'IP ex kur', 'IP skew', 'DM-SNR mean', 'DM-SNR stdev', 'DM-SNR ex kur', 'DM-SNR skew']

scatter_matrix(pulsar[features])
sns.pairplot(data = pulsar, hue = 'target_class')

X = pulsar.loc[:, features].values
y = pulsar.loc[:, ['target_class']].values

def standardize(data):
    scaler = StandardScaler().fit_transform(data)
    return scaler

#X = standardize(X)

"""
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = train_test_ratio)
#X_test, X_val, y_test, y_val = train_test_split(X_test,y_test,test_size = test_val_ratio)
X_train = X[0:13423,:]
y_train = y[0:13423,:]
X_test = X[13424:17898,:]
y_test = y[13424:17898,:]




##############Linear Regression#####################

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_predLin = lin_model.predict(X_test)

predClassLin = y_predLin > 0.5
misClassLin = np.mean(y_test != predClassLin)
print("Misclassification rate for linear regression model: {}".format(misClassLin))

##############Logistic Regression###################

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train.ravel())
y_predLog = log_model.predict(X_test)
predClassLog = y_predLog > 0.5
y_test_bool = y_test > 0.5
misClassLog = np.mean(y_test_bool.ravel() != predClassLog)
print("Misclassification rate for logistic regression model: {}".format(misClassLog))
cnf_matrix = metrics.confusion_matrix(y_test, y_predLog)

##############LASSO regression#####################



reg = LassoCV(cv = 10, max_iter=10000)
reg.fit(X_train,y_train.ravel())
alpha = reg.alpha_

LASSO_best = Lasso(alpha=alpha)
LASSO_best.fit(X_train,y_train)

y_predLASSOLin = LASSO_best.predict(X_test)
predClassLinLASSO = y_predLASSOLin > 0.5
misClassLinLASSO = np.mean(y_test.ravel() != predClassLinLASSO)
print("Misclassification rate for LASSO fittet linear regression model: {}" .format(misClassLinLASSO))


plt.semilogx(reg.alphas_, reg.mse_path_, ":")
plt.plot(
    reg.alphas_ ,
    reg.mse_path_.mean(axis=-1),
    "k",
    label="Average across the folds",
    linewidth=2,
)   
plt.axvline(
    reg.alpha_, linestyle="--", color="k", label="Lambda: CV estimate")
plt.legend()
plt.xlabel("Lambdas")
plt.ylabel("Mean square error")
plt.title("Mean square error on each fold")
plt.axis("tight")


log_reg = LogisticRegressionCV(penalty="l1", cv=10, solver = 'liblinear', max_iter=10000)
log_reg.fit(X_train, y_train.ravel())
log_lambda = log_reg.C_
logLASSO_best = LogisticRegression(penalty='l1', C=log_lambda[0], solver = 'liblinear', max_iter = 10000)
logLASSO_best.fit(X_train, y_train.ravel())

y_predLASSOLog = logLASSO_best.predict(X_test)
predClassLogLASSO = y_predLASSOLog > 0.5
misClassLogLASSO = np.mean(y_test.ravel() != predClassLogLASSO)
print("Misclassification rate for LASSO fittet logistic regression model: {}".format(misClassLogLASSO))

logScores = np.transpose(log_reg.scores_.get(1))

plt.semilogx(log_reg.Cs_, logScores, ":")
plt.plot(
    log_reg.Cs_ ,
    logScores.mean(axis=-1),
    "k",
    label="Average across the folds",
    linewidth=2,
)   
plt.axvline(
    log_reg.C_[0], linestyle="--", color="k", label="Lambda: CV estimate")
plt.legend()
plt.xlabel("Lambdas")
plt.ylabel("Accuracy score")
plt.title("Accuracy score on each fold")
plt.axis("tight")
"""



    
"""


def pca_truePossitive(data, features, n, cols):
    data_tp = data
    
    x = data_tp.loc[:, features].values
    #y = data_tp.loc[:, ['target_class']].values
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=n)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = cols[0:n])
    finalDf = pd.concat([principalDf, data[['target_class']]], axis = 1)
    ratio = pca.explained_variance_ratio_
    return finalDf, ratio

columns = ['pc1','pc2', 'pc3', 'pc4', 'pc5', 'pc6', 'pc7', 'pc8']
pulsar_pca, pca_ratio = pca_truePossitive(pulsar, features, 8, columns)
print(pca_ratio)
   
    
"""



