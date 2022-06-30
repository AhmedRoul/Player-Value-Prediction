#!/usr/bin/env python
# coding: utf-8

# In[6]:


import time
import pickle
import datetime as dt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


def Feature_Encoder(X, cols):
    for c in cols:
        # lbl = LabelEncoder()
        # lbl.fit(list(X[c].values))
        fileName = c + ".pickle"
        # pick_in = open(fileName, 'wb')
        # pickle.dump(lbl, pick_in)
        # pick_in.close()
        with open(fileName,'rb') as file_handle:
             lbl = pickle.load(file_handle)
        X[c] = lbl.transform(list(X[c].values))
    return X


def featureScaling(X, a, b):
    X = np.array(X)
    Normalized_X = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:, i] = ((X[:, i] - min(X[:, i])) / (max(X[:, i]) - min(X[:, i]))) * (b - a) + a
    return Normalized_X


# Load players data
data = pd.read_csv('player-test-samples.csv')
# Drop the columns that contain unimportant values
c = ['id', 'name', 'full_name', 'age', 'birth_date', 'height_cm', 'weight_kgs'
    , 'preferred_foot', 'club_join_date', 'contract_end_year', 'national_team'
    , 'national_rating', 'national_team_position', 'national_jersey_number', 'tags'
    , 'club_rating', 'club_jersey_number']

data = data.drop(columns=c)

# split position between "," and make "one encoding"
# each position has column
data_positions = data['positions'].str.get_dummies(sep=',').add_prefix("pos_")
for col in data_positions.columns:
    data.insert(loc=5, value=data_positions[col], column=col)
data.drop('positions', axis=1, inplace=True)


# ----------------------------------------------
X = data.iloc[:, :-1]
#print(X.columns)
Y = data['value']
#print(Y.mean())
# ------------------
columnsNeedEncoding = ['nationality', 'work_rate', 'body_type', 'club_team', 'club_position']
X = Feature_Encoder(X, columnsNeedEncoding)
columnsNeedPlus = ['ST', 'LS', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW', 'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM'
    , 'RM', 'LWB', 'LDM', 'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']
for i in columnsNeedPlus:
    X[i] = X[i].str.replace('+', '', regex=True)
    X[i] = pd.to_numeric(X[i].str[:2]) + pd.to_numeric(X[i].str[-1])

# -------------------------------------------------
# Traits Pre-processing

data_positions = X['traits'].str.get_dummies(sep=',').add_prefix("pos_")
for col in data_positions.columns:
    X.insert(loc=5, value=data_positions[col], column=col)
X.drop('traits', axis=1, inplace=True)
international_reputation_column = X.columns.get_loc("international_reputation(1-5)")
#Adding Traits to one column
X['traits'] = X.iloc[:, 5:international_reputation_column].sum(axis=1)
X.drop(X.iloc[:, 5:international_reputation_column], axis=1 , inplace = True)
#Change column place
X.insert(1, 'traits', X.pop('traits'))

# ---------------------------------
mean=pd.read_csv("mean.csv")
for i in range(len(mean)-1):
    X[mean.loc[i,"name"]] = X[mean.loc[i,"name"]].fillna(mean.loc[i,"mean"])

Y = Y.fillna(mean.loc[len(mean)-1,"mean"])
# Get the correlation between the features
#corr = data.corr()
# Top 70% Correlation training features with the Value and plot
#top_feature = corr.index[abs(corr['value']) > 0.7]
#top_feature = top_feature.delete(-1)

#print(top_feature)

# with open('correlation.pickle', 'wb') as handle:
#     pickle.dump(top_feature, handle)
# with open('correlation.pickle', 'rb') as handle:
#     top_feature = pickle.load(handle)
# plt.subplots(figsize=(12, 8))
# top_corr = data[top_feature].corr()
# sns.heatmap(top_corr, annot=True)
# plt.show()
# Feature Scaling and deleting the value column
# top_feature = top_feature.delete(-1)
# X = X[top_feature]
arr = [X['wage'],X['release_clause_euro']]
for i in X:
    if i == 'wage' or i == 'release_clause_euro':
        continue
    else:
        X = X.drop(columns = i)

X = featureScaling(X,0,1)
# print(X.shape)
# print(Y.shape)
#print(X.columns)
#print(X)
# print(len(X))
# print(len(Y))

#---------------------------
#multiple regression

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,shuffle=True,random_state=10)
# model = linear_model.LinearRegression()
# start_time1 = time.time()
# model.fit(X,Y)
# with open('multiple.pickle', 'wb') as handle:
#     pickle.dump(model, handle)
with open('multiple.pickle', 'rb') as handle:
    model = pickle.load(handle)
# stop_time1 = time.time()

prediction= model.predict(X)
print('Mean Square Error for multiple regression model :', metrics.mean_squared_error(np.asarray(Y), prediction))
print('Accuracy for multiple regression model :', r2_score(Y, prediction))
# print('Training time for multiple regression model :',stop_time1-start_time1)
print('\n')

#true_value=np.asarray(y_test)[0]
#predicted_value=prediction[0]
#print('True value is : ' + str(true_value))
#print('Predicted value is : ' + str(predicted_value))
#plt
# for i in range(len(X[:][0])):
#     plt.scatter(X_train[:,i], y_train, color = 'red')
#     plt.xlabel("X {num}".format(num=i), fontsize = 20)
#     plt.ylabel("Value", fontsize = 20)
#     plt.plot(X_test[:,i], prediction, color = 'green')
#     plt.show()
# print('-------------------------------------------------')

#----------------------------------------------------------
#polynomial regression
poly_features = PolynomialFeatures(degree=3)
# transforms the existing features to higher degree features.
# with open('polynomialfeatures.pickle', 'wb') as handle:
#     pickle.dump(poly_features, handle)
# with open('polynomialfeatures.pickle', 'rb') as handle:
#     poly_features = pickle.load(handle)

# X = poly_features.fit_transform(X)
# # fit the transformed features to Linear Regression
# poly_model = linear_model.LinearRegression()
# # start_time2 = time.time()
# poly_model.fit(X, Y)
# with open('polynomialmodel.pickle', 'wb') as handle:
#     pickle.dump(poly_model, handle)
# with open('polynomialmodel.pickle', 'rb') as handle:
#     poly_model = pickle.load(handle)
#
#
# # stop_time2 = time.time()
# # predicting on training data-set
# y_train_predicted = poly_model.predict(X)
# y_predict=poly_model.predict(poly_features.transform(X))
# # predicting on test data-set
#
# prediction = poly_model.predict(poly_features.fit_transform(X))

# transforms the existing features to higher degree features.
X_train_poly = poly_features.fit_transform(X)

# fit the transformed features to Linear Regression
poly_model = linear_model.LinearRegression()
start_time2 = time.time()
poly_model.fit(X_train_poly, Y)

stop_time2 = time.time()
# predicting on training data-set
y_train_predicted = poly_model.predict(X_train_poly)
ypred=poly_model.predict(poly_features.transform(X))
# predicting on test data-set

prediction = poly_model.predict(poly_features.fit_transform(X))

print('Mean Square Error for polynomial regression model :', metrics.mean_squared_error(Y, prediction))
print('Accuracy for polynomial regression model :', r2_score(Y, prediction))
# print('Training time for polynomial regression model :',stop_time2-start_time2)

#true_player_value=np.asarray(y_test)[0]
#predicted_player_value=prediction[0]
#print('True value for the first player in the test set in millions is : ' + str(true_player_value))
#print('Predicted value for the first player in the test set in millions is : ' + str(predicted_player_value))

# for i in range(len(X[:][0])):
#     plt.scatter(X_train[:,i], y_train, color = 'red')
#     plt.xlabel("X {num}".format(num=i), fontsize = 20)
#     plt.ylabel("Value", fontsize = 20)
#     plt.plot(X_test[:,i], prediction, color = 'green')
#     plt.show()

