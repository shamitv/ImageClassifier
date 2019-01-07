import pandas as pd
import logging
import numpy as np
import pickle
from logging import info, error
from PIL import Image
from config.paths import train_csv_file ,  processed_image_dir , data_dir
from config.data import num_classes , image_dimension
import os.path

def getCSVDataFrame():
    info('Reading image metadata from CSV :: '+train_csv_file)
    train_data=pd.read_csv(train_csv_file)
    return train_data

logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(message)s')


def getImageArray(img_id, color):
    img_file=processed_image_dir + img_id + '_' + color + '.png'
    try:
        img = Image.open(img_file)
        ret = np.array(img)
        #print(ret.shape)
        if(ret.shape[0] != image_dimension or ret.shape[1] != image_dimension ):
            raise ValueError("Unexpected image dimensions "+ "Image ID :: "+img_id+" Dimensions :: "+str(ret.shape))
        ret=ret.reshape(image_dimension,image_dimension,1)
        return ret
    except Exception as err:
        error("Couldnot load image :: {0} Error :: {1}".format(img_file,err))
        return None


def makeLabelArray(label_string):
    img_labels_str = label_string.split(' ')
    img_label_indices = np.array(img_labels_str, dtype=np.uint8)
    # initialize train_y or test_y row with all Zeros
    img_labels= np.zeros(num_classes, dtype=np.uint8)
    # Set all values to  1 where class is present
    np.put(img_labels,img_label_indices, v=1)
    return img_labels



def getTrainingData():
    train_x=[]
    train_y=[]
    train_df = getCSVDataFrame()
    info('suffle image metadata')
    train_df = train_df.sample(frac=1, random_state=4932)
    info('Loading image data')
    num_images = train_df.shape[0]
    image_count=0;
    for x in range(0, num_images):
        #if(image_count>100):
        #    break
        # Load source image
        img_id=train_df.iloc[x].values[0]
        img_arr=getImageArray(img_id,'green')
        if(img_arr is not None):
            train_x.append(img_arr)
            # Load target labels
            img_labels_str=train_df.iloc[x].values[1]
            img_labels=makeLabelArray(img_labels_str)
            train_y.append(img_labels)
            image_count += 1
    info("Loading image data :: Done Image count :: {0}".format(image_count))
    return train_x , train_y

def getPiclePath():
    pickle_path = data_dir + '/train-data.pickle'
    return pickle_path


def getData():
    pickle_path = getPiclePath()
    info("Looking for prepared data at {0}".format(pickle_path))
    if not (os.path.isfile(pickle_path)):
        info("Pickle not found. Creating it")
        preparePicle()
    ret = pickle.load(open(pickle_path, 'rb'))
    info("Pickle lodaded")
    return ret


def preparePicle():
    train_x, train_y = getTrainingData()
    info('Loaded training data')

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    info('Converted training data to Numpy arrays')

    data = {'train_x': train_x,
            'train_y': train_y}

    pickle_path = data_dir + '/train-data.pickle'

    pickle.dump(data, open(pickle_path, 'wb'), protocol=4)

    info('Saved data to :: ' + pickle_path)


'''
Test for these errors
2019-01-07 09:55:51,794 - Couldnot load image :: C:/Users/shamit/work/kaggle/Human_Protein//data//train//processed/a3824e54-bbb4-11e8-b2ba-ac1f6b6435d0_green.png Error :: tuple index out of range
2019-01-07 09:56:00,529 - Couldnot load image :: C:/Users/shamit/work/kaggle/Human_Protein//data//train//processed/37df064e-bbc6-11e8-b2bc-ac1f6b6435d0_green.png Error :: tuple index out of range
2019-01-07 09:56:23,810 - Couldnot load image :: C:/Users/shamit/work/kaggle/Human_Protein//data//train//processed/babe5cc2-bb9e-11e8-b2b9-ac1f6b6435d0_green.png Error :: tuple index out of range


def loadProblamaticImages():
    img_id_list=['a3824e54-bbb4-11e8-b2ba-ac1f6b6435d0','37df064e-bbc6-11e8-b2bc-ac1f6b6435d0','babe5cc2-bb9e-11e8-b2b9-ac1f6b6435d0'];
    for img_id in img_id_list:
        getImageArray(img_id, 'green')


loadProblamaticImages()
'''

#getData()