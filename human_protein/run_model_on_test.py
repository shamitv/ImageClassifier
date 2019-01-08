from keras_model.model_def import focal_loss , f1, precision, recall
from config.paths import data_dir, test_results_pickle
from dataset.loadData import getData, getCSVDataFrame
from logging import info
import keras
import pickle

info('Loading data')
test_data = getData('test')
test_meta = getCSVDataFrame('test')

image_ids = test_meta.iloc[:,0];

info('Getting model definition')



model_path =  data_dir + 'model/' + 'model_v8.h5.0180-22.09val_f1-0.2758.h5'

info("Loading model from :: {0}".format(model_path))

focal_loss_fixed=focal_loss(alpha=.25, gamma=2)
model = keras.models.load_model(model_path, custom_objects={'focal_loss_fixed': focal_loss_fixed ,
                                                            'f1':f1, 'precision':precision, 'recall':recall})

test_x = test_data['test_x']
info('Predicting classes')
results=model.predict(test_x, batch_size=2, verbose=1)

pickle_data = {'test_results': results , 'image_ids' : image_ids}
info("Saving data to pickle :: {0} ".format(test_results_pickle))
pickle.dump(pickle_data,open(test_results_pickle, 'wb'), protocol=4)