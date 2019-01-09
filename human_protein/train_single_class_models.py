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

def trainForLabel(label , df, class_weights):
    y_column_name='label_'+str(label)
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
                        steps_per_epoch=num_train_steps, epochs=20,verbose=1,
                        callbacks=[checkpoint],
                        validation_steps=num_val_steps,
                        validation_data=validation_generator,
                        class_weight=class_weights)

def esimateClassWeight(df, label):
    num_images=df.shape[0]
    column_name="label_{0}".format(label)
    label_positive=df.loc[df[column_name] == 1]
    positive_count=label_positive.shape[0]
    positive_ratio=num_images/positive_count
    class_weight = {0: 1, 1: positive_ratio}
    info("Positive class weight for label {0} is {1}".format(label,positive_ratio))
    return class_weight


if __name__ == "__main__":
    df = getExpandedDataFrame('train')
    for x in range(0,28):
        label=str(x)
        info("Training for label {0}".format(label))
        class_weights = esimateClassWeight(df, label)
        trainForLabel(label,df,class_weights)
    '''
    label=str(19)
    class_weights = esimateClassWeight(df, label)
    trainForLabel(label,df,class_weights)
    '''