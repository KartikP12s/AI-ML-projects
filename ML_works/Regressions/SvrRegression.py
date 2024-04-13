import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
dataset= dataset= pd.read_csv('/Users/kartikpatel/Downloads/Position_Salaries.csv')
x_ind= dataset.iloc[:,1:-1].values
y_dep= dataset.iloc[:,-1].values
#print(x_ind)
y_dep= y_dep.reshape(len(y_dep),1)

sc_x= StandardScaler()
sc_y= StandardScaler()
x_ind=sc_x.fit_transform(x_ind)
y_dep= sc_y.fit_transform(y_dep)

#train the SVR model
regressor= SVR(kernel='rbf')
regressor.fit(x_ind,y_dep)

#predicting result
print(sc_y.inverse_transform(regressor.predict(sc_x.transform([[6.5]])).reshape(-1,1)))

plt.scatter(sc_x.inverse_transform(x_ind),sc_y.inverse_transform(y_dep), color='red')
plt.plot(sc_x.inverse_transform(x_ind), sc_y.inverse_transform(regressor.predict(x_ind).reshape(-1,1)) , color='blue')
plt.title('Truth or bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()