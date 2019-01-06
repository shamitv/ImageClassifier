import platform


if platform.system()=='Windows':
    root_dir = ''
else:
    root_dir= './'

data_dir = root_dir + '/data/'

source_image_dir = data_dir+'/train/'

processed_image_dir = source_image_dir + '/processed/'

train_csv_file = data_dir + '/train.csv'

