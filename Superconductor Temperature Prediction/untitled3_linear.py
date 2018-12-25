
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn import  preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

#   Add a Dataset   #

train_df = pd.read_csv("new_featureset.csv",delimiter=',')
X = train_df.drop(['number_of_elements','critical_temp'],axis=1)
Y = train_df['critical_temp']

"""
Remove all single line comments when not want to use k fold and comment K fold portion

"""
X_scaled = preprocessing.MinMaxScaler()
X = X_scaled.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=2)

"""
# ------------- K fold Cross Val portion Starts ------------  #
"""
"""
kf = KFold(n_splits=5)
kf.get_n_splits(X)
accuracy_res = []
prediction_df = []
count=1

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=10)
    
    model = LinearRegression()
    model.fit(X_train,y_train)
    predictions = model.predict(X_test)
    accuracy = model.score(X_test,y_test)
    accuracy_res.append(accuracy)    
    prediction_df.append(predictions.tolist())    

    print (count)
    count +=1
"""



"""
# ------------- K fold Cross Val portion Ends ------------  #
"""

model = LinearRegression()
model.fit(X_train,y_train)
print (model)

predictions = model.predict(X_test)
accuracy = model.score(X_test,y_test)

print ("accuracy is: ",accuracy*100,'%')

fig, ax = plt.subplots()
ax.scatter(y_test, predictions, edgecolors=(0, 0, 0))
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

 
#model = LinearRegression()
#model.fit(X_train,y_train)
#print (model)

#predictions = predicted.predict(X_test)
#accuracy = predicted.score(X_test,y_test)

#print ("accuracy is: ",accuracy*100,'%')

#fig, ax = plt.subplots()
#ax.scatter(Y, predicted, edgecolors=(0, 0, 0))
#ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#plt.show()






