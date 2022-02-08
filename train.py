### Import libraries
print('Import libraries')
import tensorflow as tf
from multiprocessing import Pool
import os
import warnings
import glob
import re

### Remove warnings
print('Remove warnings')
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category = DeprecationWarning)

'''
TF_CPP_MIN_LOG_LEVEL = 0 to all logs .
TF_CPP_MIN_LOG_LEVEL = 1 to filter out INFO logs 
TF_CPP_MIN_LOG_LEVEL = 2 to additionall filter out WARNING 
TF_CPP_MIN_LOG_LEVEL = 3 to additionally filter out ERROR.
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
###
### Allow gpu memory growth
print('Allow gpu memory growth')
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass
###
### Set directories
print('Set directories')
human_dir = "Training_Data/human/"
spoof_dir = "Training_Data/spoof/"

human_data = 'train_data/1/'
spoof_data = 'train_data/0/'
###
### Create data generators
print('Create data generators')
batch_size = 256
    
datagen_1 = tf.keras.preprocessing.image.ImageDataGenerator(
                              horizontal_flip=True) # preprocessing_function=swap_values

datagen_0 = tf.keras.preprocessing.image.ImageDataGenerator() # preprocessing_function=swap_values

train_generator = datagen_1.flow_from_directory(directory=r"./train_data/",
                                                  class_mode="binary",
                                                  batch_size=batch_size,
                                                  target_size=(150, 150),
                                                  color_mode="rgb",) # grayscale

valid_generator = datagen_0.flow_from_directory(directory=r"./valid_data/",
                                                  class_mode="binary",
                                                  batch_size=batch_size,
                                                  target_size=(150, 150),
                                                  color_mode="rgb",) # grayscale
###
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
### Create checkpoint callback
print('Create checkpoint callback')
checkpoint_path = 'ckpt/{epoch:02d}-{accuracy:.2f}.hdf5'

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    save_best_only=False,
    monitor='accuracy',
    mode='max',
    save_freq = 1920,
    verbose=1)
###
### Compile model
print('Compile model')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=['accuracy']) #tf.keras.metrics.BinaryAccuracy()

model.summary()
###
### Load weights and train model
print('Load weights and train model')
epochs = 500

#model.load_weights('preloaded_weights/weights_Deep4SNet.h5')

list_of_files = glob.glob('ckpt/*')
latest_file = max(list_of_files, key=os.path.getctime)
model.load_weights(latest_file)
initial_epoch = int(re.search('/(.+?)-', latest_file).group(1))

history = model.fit(x = train_generator,

                    validation_data = valid_generator,
                    verbose = 2,
                    epochs = epochs,
                    workers = 5,
                    max_queue_size = 256,
                    callbacks = [model_checkpoint_callback],
                    use_multiprocessing = True,
                    initial_epoch=initial_epoch)
###