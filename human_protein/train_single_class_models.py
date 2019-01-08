from keras_model.single_class_model import getModel, getModelFile
from dataset.loadData import getExpandedDataFrame
from logging import info
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from config.paths import processed_image_dir_train
from config.data import image_dimension
import logging
logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(message)s')

batch_size=32

def trainForLabel(label):
    y_column_name='label_'+str(label)
    df = getExpandedDataFrame('train')
    num_samples=df.shape[0]
    validation_split=0.2
    num_train_steps=int(num_samples*(1-validation_split)/batch_size)
    num_val_steps = int(num_samples * (validation_split) / batch_size)
    datagen=ImageDataGenerator(validation_split=validation_split)

    train_generator = datagen.flow_from_dataframe(dataframe=df, directory=processed_image_dir_train,
                                                  x_col="image_files", y_col=y_column_name,
                                                  class_mode="binary", subset="training",color_mode='grayscale',
                                                  target_size=(image_dimension, image_dimension), batch_size=batch_size)


    validation_generator = datagen.flow_from_dataframe(dataframe=df, directory=processed_image_dir_train,
                                                  x_col="image_files", y_col=y_column_name,
                                                  class_mode="binary", subset="validation",color_mode='grayscale',
                                                  target_size=(image_dimension, image_dimension), batch_size=batch_size)

    model = getModel()
    model_file = getModelFile(label)
    info("Saving model at {0}".format(model_file))

    checkpoint = ModelCheckpoint(model_file,
                                 monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=num_train_steps, epochs=20,verbose=2,
                        callbacks=[checkpoint],
                        validation_steps=num_val_steps,
                        validation_data=validation_generator, )


if __name__ == "__main__":
    for x in range(0,28):
        info("Training for label {0}".format(x))
        trainForLabel(str(x))