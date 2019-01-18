from keras.models import Sequential
from keras.layers import Conv2D, Activation, Dropout
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from config.paths import data_dir
from config.data import image_dimension


def getModelFile(label):
    model_dir = data_dir + '/model'
    model_version = 2
    model_path = "{0}/signle_class_label_{1}_model_v{2}.h5".format(model_dir,label,model_version)
    return model_path

def getModel():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(image_dimension, image_dimension, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.5))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    # model.add(Dropout(0.5))
    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    # model.add(Dropout(0.5))
    #model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    #model.summary()
    return  model

