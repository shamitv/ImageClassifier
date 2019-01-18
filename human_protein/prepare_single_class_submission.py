import pandas as pd
from config.paths import test_single_class_output_csv, test_output_csv
from config.data import num_classes
import numpy as np


results = pd.read_pickle(test_single_class_output_csv+'.pickle')

num_samples=len(results['images'])

print('Number of images ::' + str(num_samples))
image_labels = []
for i in range(0,num_samples):
    labels=[]
    for x in range(0,num_classes):
        key="label_{0}".format(x)
        value_list=results[key]
        value=value_list[i]
        if value==1:
            labels.append(str(x))
    label_str=" ".join(labels)
    image_labels.append(label_str)

df_src_dict={'Id':np.array(results['images']),	'Predicted': np.array(image_labels)}

#df = pd.DataFrame.from_dict(data=image_lables.items(), orient='rows' ,columns=['Id',	'Predicted'])
df = pd.DataFrame.from_dict(data=df_src_dict,orient='columns' )
print(df.head())

df.to_csv(test_output_csv,index=False)

