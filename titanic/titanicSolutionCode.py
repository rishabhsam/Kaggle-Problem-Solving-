# Library Imports 
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Functions
extractTitle = lambda s :s.split(",")[1].split(".")[0]

# Importing the data
rawData = pd.read_csv("train.csv")

# Code for cleaning and shaping the Data 
rawData['Titles'] = rawData.Name.apply(extractTitle)
rawData["Age"] = rawData.groupby("Titles")['Age'].transform(lambda x: x.fillna(x.median()))
X = train[['Pclass','Sex','Age','Titles','SibSp','Parch','Fare','Embarked']]
X['Pclass']= X.Pclass.astype('category')
X= pd.get_dummies(X)
Y = train[['Survived']]


# Finalizing the train and test data
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Training the XgBoost 
model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)

# Download the test data and pedict for it 
testRawData = pd.read_csv("test.csv")
# Code for cleaning and shaping the Data 
testRawData['Titles'] = rawData.Name.apply(extractTitle)
testRawData["Age"] = rawData.groupby("Titles")['Age'].transform(lambda x: x.fillna(x.median()))
X_Predict = testRawData[['Pclass','Sex','Age','Titles','SibSp','Parch','Fare','Embarked']]
X_Predict['Pclass']= X.Pclass.astype('category')
X_Predict= pd.get_dummies(X_Predict)

# Finding uncommon columns in between 2 train and test and hence creating new columns in train
xPredictColNames = list(X_Predict)
xTrainColNames = list(X_train)
for name in list(set(xTrainColNames)- set(xPredictColNames)):
    X_Predict[name] = 0

# Arranging the column names such that the names are all aligned according to the names in the training data
X_Predict = X_Predict[xTrainColNames]

# Creating the prediction value vector
Y_Predict = model.predict(X_Predict)
