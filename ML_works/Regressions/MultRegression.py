import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
dataset= pd.read_csv('/Users/kartikpatel/Downloads/50_Startups.csv')
x_ind= dataset.iloc[:,:-1].values
y_dep= dataset.iloc[:,-1].values
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
x_ind = np.array(ct.fit_transform(x_ind))
x_train,x_test,y_train,y_test= train_test_split(x_ind,y_dep,test_size=0.2, random_state=0)

mlr= LinearRegression()
mlr.fit(x_train,y_train)

y_pred= mlr.predict(x_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))