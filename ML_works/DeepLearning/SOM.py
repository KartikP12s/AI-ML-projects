import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone,pcolor,colorbar,plot,show
dataset= pd.read_csv('/Users/kartikpatel/Downloads/Self_Organizing_Maps/Credit_Card_Applications.csv')
X=dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

sc=MinMaxScaler(feature_range=(0,1))
X= sc.fit_transform(X)
#training the Som
som=MiniSom(x=10,y=10,input_len=15,sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

#visualizing the results
bone()
pcolor(som.distance_map().T)
colorbar()
markers =['o','s']
colors=['r','g']
for i,x in enumerate(X):
    w= som.winner(X)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]],
         markeredgecolor= colors[y[i]],
         markerfacecolor='None',
         markersize= 10,
         markeredgewidth=2)
show()

#catching the frauds
mappings=som.win_map(X)
frauds= np.concatenate((mappings[(8,1)],mappings[(6,8)]), axis=0)
frauds=sc.inverse_transform(X)