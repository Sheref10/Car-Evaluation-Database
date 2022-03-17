import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dython import nominal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
#EDA
data=pd.read_csv('car.csv')
print(data.head())
print(data.describe())
print(data.info())
print(data.isnull().sum())
aver=data.groupby('class').size()
print(aver)
aver.plot.bar(color='blue')
#plt.legend(title='Car Evaluation')
#plt.figure(1)
#plt.show()
le=LabelEncoder()
data['buying']=le.fit_transform(data['buying']).astype('str')
data['maint']=le.fit_transform(data['maint']).astype('str')
data['lug_boot']=le.fit_transform(data['lug_boot']).astype('str')
data['safety']=le.fit_transform(data['safety']).astype('str')
data['persons']=le.fit_transform(data['persons']).astype('str')
data['doors']=le.fit_transform(data['doors']).astype('str')

X=data.drop(['class'],axis=1)
y=data['class']
#nominal.associations(X,figsize=(20,10),mark_columns=True);
#plt.show()
#Modeling

print(y)
xtrain,xtest,ytrain,ytest=train_test_split(X,y,train_size=0.4,random_state=0)
classifier=RandomForestClassifier(n_estimators=10, criterion="entropy")
classifier.fit(xtrain, ytrain)

# 3. Predicting the test set result
y_pred= classifier.predict(xtest)
cm= confusion_matrix(ytest, y_pred)
print("confusion_matrix: \n",cm)
