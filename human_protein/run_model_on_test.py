from keras_model.model_def import getModel, getModelFile
from dataset.loadData import getData, getCSVDataFrame
from config.paths import test_results_pickle
from logging import info
import keras
import pickle

info('Loading data')
test_data = getData('test')
test_meta = getCSVDataFrame('test')

image_ids = test_meta.iloc[:,0];

info('Getting model definition')



model_path = getModelFile()

info("Loading model from :: {0}".format(model_path))

model = keras.models.load_model(model_path)

test_x = test_data['test_x']
info('Predicting classes')
results=model.predict(test_x, batch_size=2)

pickle_data = {'test_results': results , 'image_ids' : image_ids}
info("Saving data to pickle :: {0} ".format(test_results_pickle))
pickle.dump(pickle_data,open(test_results_pickle, 'wb'), protocol=4)