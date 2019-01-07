from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from config.paths import data_dir
from config.data import num_classes , image_dimension

def getModelFile():
    model_dir = data_dir + '/model'
    model_version = 2
    model_path = "{0}/model_v{1}.h5".format(model_dir,model_version)
    return model_path

def getModel():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(image_dimension, image_dimension, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.5))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    # model.add(Dropout(0.5))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='relu'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy','categorical_accuracy'])

    model.summary()
    return  model