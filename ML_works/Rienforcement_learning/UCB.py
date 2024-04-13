import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
dataset= pd.read_csv('/Users/kartikpatel/Downloads/Ads_CTR_Optimisation.csv')
N=10000
d=10
ads_selected=[]
numbers_of_selections=[0] * d
sums_of_rewards=[0] * d
total_reward=0
for x in range(0,N):
    max_upperbound=0
    ad=0
    for y in range(0,d):
        if numbers_of_selections[y]>0:
            average_reward= sums_of_rewards[y]/numbers_of_selections[y]
            delta_i= math.sqrt(3/2 * math.log(y+1) / numbers_of_selections[y])
            upper_bound=average_reward+delta_i
        else:
            upper_bound= 1e400
        if upper_bound>max_upperbound:
            max_upperbound=upper_bound
            ad=y
    ads_selected.append(ad)
    numbers_of_selections[ad]+=1
    sums_of_rewards[ad]+=dataset.values[x,ad]
    total_reward+=dataset.values[x,ad]

plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
