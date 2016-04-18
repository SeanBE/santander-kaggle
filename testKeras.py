import numpy as np
import pandas as pd

from pandas.core.common import array_equivalent
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

idCol = "ID"
targetCol = "TARGET"

testData = pd.read_csv('data/test.csv')
trainData = pd.read_csv('data/train.csv')

colCount = trainData.shape[1]

for column in trainData:

    if trainData[column].std() == 0:  # or len(pd.unique(..)) < 2
        trainData.drop(column, axis=1, inplace=True)
        testData.drop(column, axis=1, inplace=True)

print colCount - trainData.shape[1], 'columns removed from test/train data.'

# stackoverflow.com/questions/python-pandas-remove-duplicate-columns

def duplicate_columns(frame):
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:, i].values
            for j in range(i + 1, lcs):
                ja = vs.iloc[:, j].values
                if array_equivalent(ia, ja):
                    dups.append(cs[i])
                    break

    return dups

colCount = trainData.shape[1]

dupCols = duplicate_columns(trainData)
trainData.drop(dupCols, axis=1, inplace=True)
testData.drop(dupCols, axis=1, inplace=True)

print colCount - trainData.shape[1], 'columns removed from test/train data.'

x = trainData.drop([idCol, targetCol], axis=1)
y = trainData[targetCol]

scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)

train_X, test_X, train_y, test_y = train_test_split(x, y, train_size=0.8, random_state=0)

model = Sequential()
model.add(Dense(400, input_dim=x.shape[1]))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
model.fit(train_X, train_y, nb_epoch=100, batch_size=32, class_weight={0:0.04, 1:0.96}, validation_split=0.1)
loss,acc = model.evaluate(test_X, test_y, batch_size=32)

print loss,acc

testX = testData.drop([idCol], axis=1)
scaler.fit(testX)
testX = scaler.transform(testX)

testY = model.predict_proba(testX)

submission = pd.DataFrame({idCol: testData[idCol], targetCol: testY[:,0]})
submission.to_csv("output/keras.csv", index=False)
