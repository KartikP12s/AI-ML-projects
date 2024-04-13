import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
dataset= pd.read_csv('/Users/kartikpatel/Downloads/Salary_Data.csv')
x_ind= dataset.iloc[:,:-1].values
y_dep= dataset.iloc[:,-1].values
x_train,x_test,y_train,y_test= train_test_split(x_ind,y_dep,test_size=0.2, random_state=0)

#training the model
regressor= LinearRegression()
regressor.fit(x_train,y_train)

#predicting test results
y_pred= regressor.predict(x_test)

# visualizing the training set results
plt.scatter(x_train,y_train, color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()

# visualizing the test set results
plt.scatter(x_test,y_test, color='red')
plt.plot(x_train,regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.show()