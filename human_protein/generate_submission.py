import numpy as np
import pandas as pd
from config.paths import data_dir, test_results_pickle

results = pd.read_pickle(test_results_pickle)


num_samples=results['image_ids'].shape[0]

print(num_samples)

image_lables = {}

for i in range(0, num_samples):
    img_id=results['image_ids'][i]
    if img_id.strip()=='':
        break;
    probs=results['test_results'][i]
    probs[probs >= 0.5] = 1
    probs[probs < 0.5] = 0
    labels = np.where(probs == 1)
    arr = np.array(labels).flatten()
    str_val = " ".join(arr.astype("str"))
    image_lables[img_id]=str_val

#df = pd.DataFrame.from_dict(data=image_lables.items(), orient='rows' ,columns=['Id',	'Predicted'])
df = pd.DataFrame.from_dict(data=image_lables,orient='index')
print(df.head())
#df.columns=['Id',	'Predicted']

print(df.columns.values)