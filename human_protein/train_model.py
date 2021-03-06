from keras_model.single_class_model import getModel, getModelFile
from dataset.loadData import getData, getExpandedDataFrame
from logging import info
from keras.callbacks import ModelCheckpoint

info('Loading data')
train_data = getData('train')
info('Getting model definition')
model = getModel()

model_path = getModelFile()

info("Saving model at :: {0}".format(model_path))

batch_size=32
epochs=200

checkpoint = ModelCheckpoint(getModelFile()+'.{epoch:04d}-{val_loss:.2f}val_f1-{val_f1:.4f}.h5',
                             monitor='val_f1', verbose=1, save_best_only=True, mode='max')

train_x=train_data['train_x']
train_y=train_data['train_y']

info(train_x.shape)
info(train_y.shape)

model.fit(x=train_x, y=train_y, batch_size=batch_size, verbose=1,
          epochs=epochs, validation_split=0.2, callbacks=[checkpoint] )