import platform


if platform.system()=='Windows':
    root_dir = 'F:/mldata/Petfinder/'
else:
    root_dir= '/home/ubuntu/petfinder/'

data_dir = root_dir + '/data/'

log_dir = root_dir + '/log/'

tensorboard_log_dir = log_dir + '/tensorboard/'

source_image_dir_train = data_dir+'/train_images/'

processed_image_dir_train = source_image_dir_train + '/processed/'

source_image_dir_test = data_dir+'/test_images/'

processed_image_dir_test = source_image_dir_test + '/processed/'


train_csv_file = data_dir + '/train/train.csv'
breed_labels_csv = data_dir + '/breed_labels.csv'
test_csv_file = data_dir + '/test/test.csv'
test_results_pickle = data_dir + '/test_results.pickle'
test_output_csv = data_dir + '/test_results.csv'
test_single_class_output_csv = data_dir + '/test_results_single_class.csv'