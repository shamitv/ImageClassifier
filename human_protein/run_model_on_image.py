from PIL import Image
import numpy as np
from config.data import num_classes , image_dimension
from keras_model.model_def import focal_loss , f1, precision, recall
from config.paths import data_dir
import keras
image_file='F:/mldata/Human_Protein_Atlas/kaggle_competition/data/train/processed/3184e928-bbb0-11e8-b2ba-ac1f6b6435d0_green.png'
img = Image.open(image_file)
img_arr = np.array(img)

#img.show()
img_arr=img_arr.reshape(image_dimension,image_dimension,1)

img_arr2=img_arr.reshape(image_dimension,image_dimension)
img2 = Image.fromarray(img_arr2, 'L')

#img2.show()

test_x=np.array([img_arr])

print(test_x.shape)

model_path =  data_dir + 'model/' + 'model_v8.h5.0103-18.27val_f1-0.2443.h5'

focal_loss_fixed=focal_loss(alpha=.25, gamma=2)
model = keras.models.load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss_fixed ,
                                                            'f1':f1, 'precision':precision, 'recall':recall})



results=model.predict(test_x, batch_size=2, verbose=1)

preds=results[0]
preds[preds>=0.5] = 1
preds[preds<0.5] = 0
print(preds)

ones=np.where(preds == 1)

ones=np.array(ones).flatten()
str_val = " ".join(ones.astype("str"))

print(str_val)