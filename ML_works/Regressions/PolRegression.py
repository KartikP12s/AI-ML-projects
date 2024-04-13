import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
dataset= pd.read_csv('/Users/kartikpatel/Downloads/Position_Salaries.csv')
x_ind= dataset.iloc[:,1:-1].values
y_dep= dataset.iloc[:,-1].values

regressor1 = LinearRegression()
regressor1.fit(x_ind,y_dep)

pf= PolynomialFeatures(degree=4)
x_poly= pf.fit_transform(x_ind)

lr2=LinearRegression()
lr2.fit(x_poly,y_dep)

# plt.scatter(x_ind,y_dep, color='red')
# plt.plot(x_ind,regressor1.predict(x_ind), color='blue')
# plt.title('Truth or bluff (Linear Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

# plt.scatter(x_ind,y_dep, color='red')
# plt.plot(x_ind,lr2.predict(x_poly), color='blue')
# plt.title('Truth or bluff (Polynomial Regression)')
# plt.xlabel('Position level')
# plt.ylabel('Salary')
# plt.show()

print(regressor1.predict([[6.5]]))
print(lr2.predict(pf.fit_transform([[6.5]])))

