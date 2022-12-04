import os
import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt
import tensorflow as tf


model = tf.keras.models.load_model('digits.model')

for x in range(1,6):
    img = cv.imread(f'{x}.png')[:,:,0]
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(f'The result is : {np.argmax(prediction)}' )
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()

#<3