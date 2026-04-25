import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("model.h5")

classes = ['No_DR','Mild','Moderate','Severe','Proliferative_DR']

img = cv2.imread("test.jpg")
img = cv2.resize(img,(224,224))
img = img/255.0
img = np.reshape(img,(1,224,224,3))

pred = model.predict(img)
print("Prediction:", classes[np.argmax(pred)])