import pandas as pd
import logging
import numpy as np
from logging import info
from PIL import Image
from config.paths import train_csv_file , source_image_dir , processed_image_dir


def getCSVDataFrame():
    info('Reading image metadata from CSV :: '+train_csv_file)
    train_data=pd.read_csv(train_csv_file)
    return train_data

logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(message)s')


def getImageArray(img_id, color):
    img_file=processed_image_dir + img_id + '_' + color + '.png'
    img = Image.open(img_file)
    ret = np.array(img)
    return ret


def getTrainingData():
    train_x=[]
    train_df = getCSVDataFrame()
    info('suffle image metadata')
    train_df = train_df.sample(frac=1, random_state=4932)
    info('Loading image data')
    num_images = train_df.shape[0]
    for x in range(0, num_images):
        img_id=train_df.iloc[x].values[0]
        img_labels_str=train_df.iloc[x].values[1]
        img_labels=img_labels_str.split(' ')
        img_labels=np.array(img_labels, dtype=np.uint8)
        img_arr=getImageArray(img_id,'green')
        train_x.append(img_arr)
    info('Loading image data :: Done')
    return train_x


getTrainingData()