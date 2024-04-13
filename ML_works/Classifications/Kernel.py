import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
dataset= pd.read_csv('/Users/kartikpatel/Downloads/Social_Network_Ads.csv')
x_ind=dataset.iloc[:,:-1].values
y_dep=dataset.iloc[:,-1].values

x_train,x_test,y_train,y_test= train_test_split(x_ind,y_dep, test_size= 0.25, random_state=0)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

lr= SVC(kernel='rbf', random_state=0)
lr.fit(x_train,y_train)
print(lr.predict(sc.transform([[30,87000]])))

y_pred= lr.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)), 1))

cm= confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))
