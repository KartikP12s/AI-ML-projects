import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
dataset= dataset= pd.read_csv('/Users/kartikpatel/Downloads/Position_Salaries.csv')
x_ind= dataset.iloc[:,1:-1].values
y_dep= dataset.iloc[:,-1].values

regressor= DecisionTreeRegressor(random_state=0)
regressor.fit(x_ind,y_dep)

print(regressor.predict([[6.5]]))
x_grid=np.arange(min(x_ind),max(x_ind),0.1)
x_grid= x_grid.reshape(len(x_grid),1)
plt.scatter(x_ind,y_dep, color='red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')
plt.title('Truth or bluff (Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()