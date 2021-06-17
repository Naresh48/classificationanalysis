from pandas import read_csv
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

data4719=read_csv("path that contains dataset")
print(data4719.isnull().sum())

#labeling characters to numbers
y=data4719.iloc[:,7]
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

z=data4719['Job']
print(data4719['Job'].unique())
encoder.fit(z)
encoded_Z = encoder.transform(z)

s=data4719['Marital Status']
print(data4719['Marital Status'].unique())
encoder.fit(s)
encoded_S = encoder.transform(s)

a=data4719['Response']
print(data4719['Response'].unique())
encoder.fit(a)
encoded_A = encoder.transform(a)

d=data4719['Education']
print(data4719['Education'].unique())
encoder.fit(d)
encoded_D = encoder.transform(d)

f=data4719['housing']
print(data4719['housing'].unique())
encoder.fit(f)
encoded_F = encoder.transform(f)

G=data4719['Age']
H=data4719['balance']

x=pd.DataFrame({
        'job':encoded_Z,
        'marital status':encoded_S,
        'response':encoded_A,
        'education':encoded_D,
        'housing':encoded_F,
        'age':G,
        'balance':H})


x_train,x_test,y_train,y_test=train_test_split(x,encoded_Y,test_size=0.2,random_state=0)

LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train,y_train)
y_pred=LR.predict(x_test)
round(LR.score(x_test,y_test), 4)
print(confusion_matrix(y_test,y_pred))
#accuracy=75.82%

q=pd.DataFrame({
        'job':[5,7],
        'marital status':[1,1],
        'response':[1,1],
        'education':[1,2],
        'housing':[0,0],
        'age':[59,31],
        'balance':[49,90]},
        index=[1,2])
indp_pred=LR.predict(q)

#support vector machine algorithm
SVM = svm.SVC(decision_function_shape="ovo").fit(x_train,y_train)
y_pred1=SVM.predict(x_test)
round(SVM.score(x_test, y_test), 4)
print(confusion_matrix(y_test,y_pred1))#for confusion matrix
#accuracy=76.10%

#randomforest algorithm
RF = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0).fit(x_train,y_train)
y_pred2=RF.predict(x_test)
round(RF.score(x_test, y_test), 4)
print(confusion_matrix(y_test,y_pred2))
#accuracy=76.37%

