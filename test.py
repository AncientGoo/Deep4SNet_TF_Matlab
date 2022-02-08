import tensorflow as tf
import glob
import os
import re
import numpy as np

test_dir = 'test_data/t/'

### Create model
print('Create model')
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(150, 150, 3)))
model.add(tf.keras.layers.MaxPool2D(2, 2))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=64, activation="relu"))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))
###
### Compile model
print('Compile model')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy']) #tf.keras.metrics.BinaryAccuracy()

list_of_files = glob.glob('ckpt/*')
latest_file = max(list_of_files, key=os.path.getctime)
model.load_weights(latest_file)
###
### Create test data generator
batch_size = 256

img_names = os.listdir('test_data/t/')

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

img_names.sort(key=natural_keys)

def test_gen(test_dir, img_names):
    for img in img_names:
        image = tf.keras.utils.load_img(test_dir+img, color_mode='rgb', target_size=(150,150))
        yield np.expand_dims(image, axis=0)

test_generator = test_gen(test_dir, img_names)
###
### Predict and save                                              
prediction = model.predict(test_generator)


wav_names = os.listdir('Testing_Data/')
wav_names.sort(key=natural_keys)

f = open('test.txt', 'w')
for i in range(len(wav_names)):
    f.write(wav_names[i]+', '+str(prediction[i][0])+'\n')
f.close()