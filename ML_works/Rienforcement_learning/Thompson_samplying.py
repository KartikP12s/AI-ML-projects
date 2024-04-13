import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
dataset= pd.read_csv('/Users/kartikpatel/Downloads/Ads_CTR_Optimisation.csv')
N=500
d=10
ads_selected=[]
number_of_rewards_1= [0]*d
number_of_rewards_0= [0]*d
total_reward=0
for i in range(0,N):
    ad=0
    max_random=0
    for j in range(0,d):
        random_beta= random.betavariate(number_of_rewards_1[j]+1 , number_of_rewards_0[j]+1)
        if (random_beta>max_random):
            max_random=random_beta
            ad=j
    ads_selected.append(ad)
    reward= dataset.values[i,ad]
    if reward ==1:
        number_of_rewards_1[ad]+=1
    else:
        number_of_rewards_0[ad]+=1
    total_reward+=reward

plt.hist(ads_selected)
plt.title('Histogram of ads selection')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
