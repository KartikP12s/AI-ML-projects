import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
dataset= pd.read_csv('/Users/kartikpatel/Downloads/Data.csv')
x_ind= dataset.iloc[:,:-1].values
y_dep= dataset.iloc[:,-1].values
imputer.fit(x_ind[:,1:3]) 
x_ind[:,1:3] = imputer.transform(x_ind[:,1:3])

# encoding country names into vectors of [1,0,0],[0,1,0],[0,0,1]
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x_ind = np.array(ct.fit_transform(x_ind))
#print(x_ind)

#labeling the dependent variable with 0s and 1s for no and yes
lp = LabelEncoder()
y_dep=lp.fit_transform(y_dep)
#print(y_dep)

#splitting the dataset into training and test sets:
x_train,x_test,y_train,y_test= train_test_split(x_ind,y_dep,test_size=0.2, random_state=1)
# print (x_train)
# print (x_test)

#feature scaling
sc=StandardScaler()
x_train[:,3:]=sc.fit_transform(x_train[:,3:])
x_test[:,3:]= sc.transform(x_test[:,3:])
print(x_train)
print(x_test)