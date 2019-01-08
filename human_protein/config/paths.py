import platform


if platform.system()=='Windows':
    root_dir = 'F:/mldata/Human_Protein_Atlas/kaggle_competition/'
else:
    root_dir= '/home/ubuntu/hpa/'

data_dir = root_dir + '/data/'

source_image_dir_train = data_dir+'/train/'
label_specific_train_dir = source_image_dir_train+'/label_train/'


processed_image_dir_train = source_image_dir_train + '/processed/'

source_image_dir_test = data_dir+'/test/'

processed_image_dir_test = source_image_dir_test + '/processed/'


train_csv_file = data_dir + '/train.csv'

test_csv_file = data_dir + '/test.csv'
test_results_pickle = data_dir + '/test_results.pickle'
test_output_csv = data_dir + '/test_results.csv'