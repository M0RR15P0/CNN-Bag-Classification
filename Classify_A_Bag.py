# Import os
import os, sys

# Importing the Keras libraries and packages
from keras.models import model_from_json
from keras.preprocessing import image

# Import PIL
from PIL import Image

# Importing NumPy
import numpy as np

# load json and create model
json_file = open('classifier.json', 'r')
loaded_classifier_json = json_file.read()
json_file.close()
loaded_classifier = model_from_json(loaded_classifier_json)

# load weights into new classifier
loaded_classifier.load_weights("classifier.h5")
print("Loaded classifier from disk")

# Classify a Bag
test_image = image.load_img("C:/Users/Morris/Documents/Charles & Keith Interview/test_image.jpg", target_size = (64, 64))# Change the directory to your photo
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_classifier.predict(test_image)

if result[0][0] == 1:
	prediction = 'A'
elif result[0][1] == 1:
	prediction = 'B'	
else:
	prediction = 'C'
print("Prediction: "+prediction)

