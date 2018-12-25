# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 08:00:42 2018

@author: krutharth
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import  preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

train_df = pd.read_csv("new_featureset.csv",delimiter=',')
X_train = train_df.drop(['number_of_elements','critical_temp'],axis=1)
Y = train_df['critical_temp']

X_scaled = preprocessing.MinMaxScaler()
X = X_scaled.fit_transform(X_train)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


model = Ridge(alpha=0.05)
model.fit(X_train,y_train)

predictions = model.predict(X_test)
accuracy = model.score(X_test,y_test)

print ("accuracy is: ",accuracy*100,'%')


fig, ax = plt.subplots()
ax.scatter(y_test, predictions, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()




















