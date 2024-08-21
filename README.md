# Machine-Learning
Project for CS 622 Intro to Machine Learnig course


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import log_loss, f1_score
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#Load dataset from scikit-learn dataset library
irisTemp = pd.read_csv('Iris.csv')
#taking data of only 2 classes setosa and virginica
irisSetosa = irisTemp[irisTemp.iloc[:,5]=='Iris-setosa']
irisVirginica = irisTemp[irisTemp.iloc[:,5]=='Iris-virginica']
#concat the data extracted from Iris csv file
irisDataSet=pd.concat([irisSetosa,irisVirginica])

#extract the target data
irisDataTarget = irisDataSet.drop(columns=['Id','SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm'])

#split the data as 80% for train and 20% for test dataset
X_train, X_test, yTempTrain, yTempTest = train_test_split(irisDataSet, irisDataTarget, test_size=0.2)

# encTargetoding the string species target class name from the dataset to 0 or 1
y_Temp_Test = yTempTest['Species'].values
encTarget = LabelEncoder()
label_encTargetoder_test = encTarget.fit(y_Temp_Test)
y_test = label_encTargetoder_test.transform(y_Temp_Test)

yTempTrain = yTempTrain['Species'].values
label_encTargetoder = encTarget.fit(yTempTrain)
y_train = label_encTargetoder.transform(yTempTrain)

#getting the values needed only by dropping the unnessary fields
X_train = X_train.drop(columns=['Id', 'Species']).values
X_test = X_test.drop(columns=['Id', 'Species']).values


#Create a Gaussian Classifier
gnb = GaussianNB()
#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
yPred = gnb.predict(X_test)

yPredTrain = gnb.predict(X_train)

# Model Accuracy, how often is the classifier correct?
print("Test dataset accuracy: {:.2f}".format(metrics.accuracy_score(y_test, yPred)))
print("Training dataset accuracy: {:.2f}".format(metrics.accuracy_score(y_train, yPredTrain)))

#test
tnTest, fpTest, fnTest, tpTest = confusion_matrix(y_test, yPred).ravel()
specificityTest = tnTest / (tnTest+fpTest)
print("Test specificity: {:.2f}".format(specificityTest))
f1Test=f1_score(y_test, yPred)
print("Test F1 score: {:.2f}".format(f1Test))
sensitivityTest = tpTest / (tpTest + fnTest)
print("Test sensitivity (recall): {:.2f}".format(sensitivityTest))


#train
tnTrain, fpTrain, fnTrain, tpTrain = confusion_matrix(y_train, yPredTrain).ravel()
specificityTrain = tnTrain / (tnTrain+fpTrain)
print("Train specificity: {:.2f}".format(specificityTrain))
f1Train=f1_score(y_train, yPredTrain)
print("Train F1 score: {:.2f}".format(f1Train))
sensitivityTrain = tpTrain / (tpTrain + fnTrain)
print("Train sensitivity (recall): {:.2f}".format(sensitivityTest))


#confusion matrix using data frame for test dataset
df = pd.DataFrame(metrics.confusion_matrix(y_test,yPred), index=['setosa','virginica'], columns=['setosa','virginica'])
print ("confusion matrix test: \n",df)
#confusion matrix using data frame for training dataset
df_T = pd.DataFrame(metrics.confusion_matrix(y_train,yPredTrain), index=['setosa','virginica'], columns=['setosa','virginica'])
print ("confusion matrix training: \n",df_T)

print("\n Test classification_report")
print(metrics.classification_report(y_test, yPred))

print("\n Training classification_report")
print(metrics.classification_report(y_train, yPredTrain))

#log loss for test dataset
pred_test = gnb.predict_proba(X_test)
print("\nLog loss for test dataset: {:.2f}".format(log_loss(y_test, pred_test)))

#log loss for training dataset
pred_train = gnb.predict_proba(X_train)
print("Log loss for training dataset: {:.2f}".format(log_loss(y_train, pred_train)))

