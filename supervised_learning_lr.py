#to detect breast cancer
#ordinary least square method
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
dataset=load_breast_cancer()
X=dataset.data
Y=dataset.target
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=72101)
pred=LinearRegression()
pred.fit(X_train,Y_train)
Y_pred=pred.predict(X_test)
df=pd.DataFrame({'PREDICTED':Y_pred,'OBSERVED':Y_test})
df.head()
##estimating the accuracy of our model
a=mean_squared_error(Y_test, Y_pred)
b=r2_score(Y_test, Y_pred)
print(a)
print(b)

pred.coef_
pred.intercept_
