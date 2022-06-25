import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
import seaborn as sns
plt.rcParams['figure.figsize']=(12,8)
data1=pd.read_csv('../input/bike-sharing/bike_sharing_data.txt')
data1.head()
data1.info

x=data1.iloc[:,0]
y=data1.iloc[:,1]
plt.scatter(x,y)
plt.show()
m=0
c=0
L=0.0001
epochs=1000
n=float(len(x))
for i in  range(len(x)):
    y_pred=m*x+c
    D_m=(-2/n)*sum(x*(y-y_pred))
    D_c=(-2/n)*sum(y-y_pred)
    m=m-L*D_m
    c=c-L*D_c
print(m,c)     
y_pred=m*x+c
plt.scatter(x,y)
plt.plot(x,y_pred)
