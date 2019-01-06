import pandas as pd
import logging
from logging import info
import shutil
import os
from dataset.loadData import getCSVDataFrame
from config.paths import train_csv_file , source_image_dir , processed_image_dir
logging.basicConfig(level=logging.DEBUG,  format='%(asctime)s - %(message)s')


def preProcessImages():
    train_data=getCSVDataFrame()
    num_images=train_data.shape[0]
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

preProcessImages()