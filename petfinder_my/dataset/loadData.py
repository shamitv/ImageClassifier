import pandas as pd
import logging
from logging import info
from config.paths import train_csv_file , test_csv_file , breed_labels_csv
logging.basicConfig(level=logging.INFO,  format='%(asctime)s - %(message)s')

def getCSVDataFrame(data_source='train'):
    if(data_source=='train'):
        csv_file=train_csv_file
    else:
        csv_file=test_csv_file
    info('Reading data from CSV :: '+csv_file)
    train_data=pd.read_csv(csv_file)

    return train_data


def getBreeds():
    breed_df = pd.read_csv(breed_labels_csv)
    
    return breed_df


if __name__ == "__main__":
    print('Loading data')