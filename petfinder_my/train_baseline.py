import pandas as pd


from keras_model.model_def import getModel
from logging import info

from dataset.pre_process import getTrainDF , convertColumnToOneHot

info('Getting data')

df=getTrainDF()

info('Got data')



train_X = df.drop(columns=['AdoptionSpeed'])
train_y = df[['AdoptionSpeed']]

train_y=convertColumnToOneHot(train_y,'AdoptionSpeed',[0,1,2,3,4])
num_classes=5
num_columns=train_X.shape[1]

model=getModel(num_classes,num_columns)
model.fit(train_X, train_y, validation_split=0.2, epochs=30, verbose=2)