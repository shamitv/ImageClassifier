import pandas as pd
import logging
import numpy as np
import pickle
from logging import info, error
from PIL import Image
from config.paths import train_csv_file , test_csv_file ,  processed_image_dir_train, processed_image_dir_test , data_dir
from config.data import num_classes , image_dimension , train_color
import os.path


def getCSVDataFrame(data_source='train'):
    if(data_source=='train'):
        csv_file=train_csv_file
    else:
        csv_file=test_csv_file
    info('Reading image metadata from CSV :: '+csv_file)
    train_data=pd.read_csv(csv_file)
    return train_data

logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(message)s')


def getImageArray(img_id, color, data_source='train'):
    if(data_source=='train'):
        image_dir=processed_image_dir_train
    else:
        image_dir=processed_image_dir_test

    img_file=image_dir + img_id + '_' + color + '.png'
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
    img_labels_str = str(label_string).split(' ')
    img_label_indices = np.array(img_labels_str, dtype=np.uint8)
    # initialize train_y or test_y row with all Zeros
    img_labels= np.zeros(num_classes, dtype=np.uint8)
    # Set all values to  1 where class is present
    np.put(img_labels,img_label_indices, v=1)
    return img_labels



def getImageData(data_source='train'):
    train_x=[]
    train_y=[]
    train_df = getCSVDataFrame(data_source)
    if(data_source=='train'):
        info('suffle image metadata')
        train_df = train_df.sample(frac=1, random_state=4932)
    info("Loading image data for {0}".format(data_source))
    num_images = train_df.shape[0]
    image_count=0;
    for x in range(0, num_images):
        #if(image_count>100):
        #    break
        # Load source image
        img_id=train_df.iloc[x].values[0]
        img_arr=getImageArray(img_id,'green', data_source)
        if(img_arr is not None):
            train_x.append(img_arr)
            # Load target labels
            img_labels_str=train_df.iloc[x].values[1]
            img_labels=makeLabelArray(img_labels_str)
            train_y.append(img_labels)
            image_count += 1
    info("Loading image data :: Done Image count :: {0}".format(image_count))
    return train_x , train_y

def getPiclePath(data_source='train'):
    if(data_source=='train'):
        pickle_name='/train-data.pickle'
    else:
        pickle_name='/test-data.pickle'
    pickle_path = data_dir + pickle_name
    return pickle_path


def getData(data_source='train'):
    pickle_path = getPiclePath(data_source)
    info("Looking for prepared data at {0}".format(pickle_path))
    if not (os.path.isfile(pickle_path)):
        info("Pickle not found. Creating it")
        if (data_source == 'train'):
            preparePicle()
        else:
            preparePicleTest()
    ret = pickle.load(open(pickle_path, 'rb'))
    info("Pickle lodaded")
    return ret


def preparePicle():
    train_x, train_y = getImageData('train')
    info('Loaded training data')

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    info('Converted training data to Numpy arrays')

    data = {'train_x': train_x,
            'train_y': train_y}

    pickle_path = getPiclePath('train')

    pickle.dump(data, open(pickle_path, 'wb'), protocol=4)

    info('Saved data to :: ' + pickle_path)

def preparePicleTest():
    test_x, test_y = getImageData('test')
    info('Loaded test data')

    test_x = np.array(test_x)
    test_y = np.array(test_y)

    info('Converted training data to Numpy arrays')

    data = {'test_x': test_x,
            'test_y': test_y}

    pickle_path = getPiclePath('test')

    pickle.dump(data, open(pickle_path, 'wb'), protocol=4)

    info('Saved data to :: ' + pickle_path)

def getExpandedDataFrame(data_source='train'):
    meta_df = getCSVDataFrame(data_source)
    num_images = meta_df.shape[0]
    data_dict = {
        'images': [],
        'image_files': []
    }
    for x in range(0, 28):
        key = "label_{0}".format(x)
        data_dict[key] = []
    for x in range(0, num_images):
        img_id = meta_df.iloc[x].values[0]
        labels_str = meta_df.iloc[x].values[1]
        labels = labels_str.split()
        filename=img_id+'_'+train_color+'.png'
        data_dict['images'].append(img_id)
        data_dict['image_files'].append(filename)
        labels = set(labels)
        for y in range(0, 28):
            label_key = "label_{0}".format(y)
            label_list=data_dict[label_key]
            if str(y) in labels:
                label_list.append(1)
            else:
                label_list.append(0)
    expanded_df=pd.DataFrame.from_dict(data=data_dict,orient='columns' )
    return expanded_df
if __name__ == "__main__":
    #getData('test')
    df=getExpandedDataFrame('train')
    print(df.head(4))