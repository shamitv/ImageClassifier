from train_mnist_model import csv_test_data , csv_submission , model_file, image_size
import pandas as pd
import numpy as np
import keras

test_data=pd.read_csv(csv_test_data)

num_images=test_data.shape[0]

print(num_images)
images=[]

for x in range(0, num_images):
    img_arr=test_data.iloc[x,0:].values.reshape(image_size,image_size).astype(dtype='uint8')
    images.append(img_arr)

test_x=np.array(images)

test_x=test_x.reshape(num_images,image_size,image_size,1)

model = keras.models.load_model(model_file)

results=model.predict(test_x, batch_size=64, verbose=2)

print(results.shape)

images_ids=[]
labels=[]
for x in range(0,num_images):
    probs=results[x]
    label=probs.argmax()
    images_ids.append(x+1)
    labels.append(label)

#ImageId,ImageId

df_src_dict={'ImageId':np.array(images_ids),	'Label': np.array(labels)}
df = pd.DataFrame.from_dict(data=df_src_dict,orient='columns' )
print(df.head())
df.to_csv(csv_submission,index=False)
print("done, saved submission at "+csv_submission)
