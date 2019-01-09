from config.paths import test_single_class_output_csv
from keras_model.single_class_model import getModelFile
from config.data import num_classes , train_color
from dataset.loadData import getCSVDataFrame , getImageArray
import numpy as np
import keras
import pandas as pd
from logging import info

def loadSingleClassModels(max_label):
    models={}
    for x in range(0, max_label + 1):
        key = "label_{0}".format(x)
        models[key]=keras.models.load_model(getModelFile(str(x)))
    return models


max_label=27
meta_df = getCSVDataFrame('test')
num_images = meta_df.shape[0]
data_dict = {
    'images': []
}

models=loadSingleClassModels(max_label)

for x in range(0, max_label+1):
    key = "label_{0}".format(x)
    data_dict[key] = []
num_images = meta_df.shape[0]
for i in range (0,num_images):
    img_id = meta_df.iloc[i].values[0]
    data_dict['images'].append(img_id)
    #if i > 200:
    #   break

for x in range(0, 20):
    key = "label_{0}".format(x)
    info("Processing {0}".format(key))
    model=models[key]
    image_list=data_dict['images']
    for img in image_list:
        img_x = [getImageArray(img_id=img, color=train_color, data_source='test')]
        img_x = np.array(img_x)
        img_class=model.predict_classes(img_x).flatten()[0]
        data_dict[key].append(img_class)

output_df = pd.DataFrame.from_dict(data=data_dict,orient='columns' )
output_df.to_csv(test_single_class_output_csv ,index=False)