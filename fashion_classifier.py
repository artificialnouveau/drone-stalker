import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# load model
model = tf.keras.models.load_model('fashion_item_classifier.h5')

# list of fashion item classes as per Fashion MNIST labels
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# load image
image = Image.open('test.jpg').convert('L')  # replace 'test.jpg' with your image file
image = image.resize((28, 28))

# preprocess image
image = img_to_array(image)
image = image.reshape(1, 28, 28, 1)
image = image / 255.0

# make prediction
prediction = model.predict(image)
predicted_class = classes[np.argmax(prediction)]

print('Predicted fashion item:', predicted_class)
