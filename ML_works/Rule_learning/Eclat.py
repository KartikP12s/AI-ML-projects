import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from apyori import apriori
dataset= pd.read_csv('/Users/kartikpatel/Downloads/Market_Basket_Optimisation.csv',header=None)
transactions=[]
for x in range(0,7501):
    transactions.append([str(dataset.values[x,j]) for j in range(0,20)])
rules= apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift= 3, min_length=2, max_length=2)
results= list(rules)

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])
resultsinDataFrame= resultsinDataFrame.nlargest(n=10,columns='Support')
print(resultsinDataFrame)
