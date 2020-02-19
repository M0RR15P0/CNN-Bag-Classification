# Import os
import os, sys

# Import tensorflow
import tensorflow as tf 

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.preprocessing import image

# Import PIL
from PIL import Image

# Initialize the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Add a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))

# Compile the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fit the CNN to the images and Data Augmentation
from keras.preprocessing.image import ImageDataGenerator
from scipy import ndimage
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
validation_datagen = ImageDataGenerator(rescale = 1./255)

# Model Training
training_set = train_datagen.flow_from_directory('Summer 2018 Pics/training_set',# Change the directory to your training set folder
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical')
validation_set = validation_datagen.flow_from_directory('Summer 2018 Pics/validation_set',# Change the directory to your validation set folder
target_size = (64, 64),
batch_size = 10,
class_mode = 'categorical')
classifier.fit_generator(training_set,
steps_per_epoch = 3,
epochs = 200,
validation_data = validation_set,
validation_steps = 1)

# Make new predictions on test set
import numpy as np
for dirname, dirnames, filenames in os.walk('Summer 2018 Pics/test_set/'):# Change the directory to your test set folder 
	for subdirname in dirnames:
		subdirectories = os.path.join(dirname, subdirname)
	for filename in filenames:
		print("Photo to Test: "+os.path.join(dirname, filename))
		test_image = image.load_img(os.path.join("C:/Users/Morris/Documents/Charles & Keith Interview",os.path.join(dirname, filename)), target_size = (64, 64))# Change the directory to your test images
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		result = classifier.predict(test_image)
		training_set.class_indices
		if result[0][0] == 1:
			prediction = 'A'
		elif result[0][1] == 1:
			prediction = 'B'	
		else:
			prediction = 'C'
		print("Prediction: "+prediction)

# Serialize model to JSON
classifier_json = classifier.to_json()
with open("classifier.json", "w") as classifier_file:
    classifier_file.write(classifier_json)

# Serialize weights to HDF5
classifier.save_weights("classifier.h5")
print("Saved classifier to disk")