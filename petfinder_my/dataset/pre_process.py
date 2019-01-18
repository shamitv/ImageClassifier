import pandas as pd
import logging
from logging import info
import shutil
import os
import numpy as np
from sklearn.preprocessing import  OneHotEncoder
from dataset.loadData import getCSVDataFrame, getBreeds
from config.paths import train_csv_file , source_image_dir_train , processed_image_dir_train, \
    source_image_dir_test , processed_image_dir_test
logging.basicConfig(level=logging.DEBUG,  format='%(asctime)s - %(message)s')


def convertColumnToOneHot(df, col_name, label_values):
    label_values.sort()
    ColumnList = df[col_name].tolist()
    Col_arr = np.array(ColumnList)
    Col_arr = Col_arr.reshape(len(Col_arr), 1)
    enc = OneHotEncoder(handle_unknown='ignore', categories=[label_values], sparse=False)
    onehot_encoded = enc.fit_transform(Col_arr)
    num_cols = onehot_encoded.shape[1]
    one_hot_col_names = []
    for n in range(num_cols):
        col_name_one_hot = col_name + '_one_hot_' + str(n + 1)
        one_hot_col_names.append(col_name_one_hot)
    one_hot_df = pd.DataFrame(onehot_encoded, columns=one_hot_col_names)
    df = pd.concat([df, one_hot_df], axis=1)
    df = df.drop( columns=[col_name])
    return df

def getTrainDF():
    src_df = getCSVDataFrame('train')
    breeds = getBreeds()
    columns_baseline = ['AdoptionSpeed', 'Type', 'Breed1', 'Breed2', 'Gender', 'Color1', 'Color2', 'Color3',
                        'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'Quantity',
                        'Fee', 'VideoAmt', 'PhotoAmt']
    df = src_df[columns_baseline].copy()
    breed_Ids = breeds["BreedID"].tolist()
    df = convertColumnToOneHot(df, "Breed1", breed_Ids)
    df = convertColumnToOneHot(df, "Breed2", breed_Ids)
    return df

def preProcessImages(data_source):
    train_data=getCSVDataFrame(data_source)
    num_images=train_data.shape[0]
    if(data_source=='train'):
        source_image_dir=source_image_dir_train
        processed_image_dir=processed_image_dir_train
    else:
        source_image_dir=source_image_dir_test
        processed_image_dir=processed_image_dir_test
    info('Number of images :: '+ str(num_images))
    info('Image copy, src = ' + source_image_dir + ' dest = ' +processed_image_dir)
    os.makedirs(processed_image_dir, exist_ok=True)
    for x in range(0, num_images):
        img_id=train_data.iloc[x].values[0]
        filename = img_id + '_green.png'
        src_file = source_image_dir + filename
        dest_file = processed_image_dir + filename
        shutil.copyfile(src_file,dest_file)
    info('Image copy :: Complete')

#preProcessImages('test')