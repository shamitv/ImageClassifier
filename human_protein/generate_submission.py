import numpy as np
import pandas as pd
from config.paths import data_dir, test_results_pickle , test_output_csv

results = pd.read_pickle(test_results_pickle)


num_samples=results['image_ids'].shape[0]

print(num_samples)

image_lables = []

for i in range(0, num_samples):
    img_id=results['image_ids'][i]
    probs=results['test_results'][i]
    probs[probs >= 0.5] = 1
    probs[probs < 0.5] = 0
    labels = np.where(probs == 1)
    arr = np.array(labels).flatten()
    if arr.shape[0] > 1:
        arr = map(str, arr)
        arr = sorted(arr)
        str_val = " ".join(arr)
    else:
        str_val=''
    image_lables.append(str_val)

df_src_dict={'Id':np.array(results['image_ids']),	'Predicted': np.array(image_lables)}

#df = pd.DataFrame.from_dict(data=image_lables.items(), orient='rows' ,columns=['Id',	'Predicted'])
df = pd.DataFrame.from_dict(data=df_src_dict,orient='columns' )
print(df.head())

df.to_csv(test_output_csv,index=False)

