filename = "/usr/local/lib/python3.7/dist-packages/keras_vggface/models.py"
text = open(filename).read()
open(filename, "w+").write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

import streamlit as st
import cv2
from PIL import Image
from keras.models import load_model
import tensorflow as tf
import tensorflow_probability as tfp
import keras.utils as image
from keras_vggface.utils import preprocess_input
import numpy as np

def pearson_corr(y_test, y_pred):
  corr = tfp.stats.correlation(y_test, y_pred)
  return corr

model = tf.keras.models.load_model('My_model_vgg16.h5', custom_objects={'pearson_corr': pearson_corr})


def predict_class(image, model):
  img = image.copy()
  img = cv2.resize(img, (224, 224))
  img = np.array(img).astype(np.float32)
  img = np.expand_dims(img, axis = 0)
  img = preprocess_input(img, version=2)
  prediction = model.predict(img)[0][0]
  return prediction

st.title('**BMI prediction 📷**')
st.write('😊')
file_image = st.camera_input(label = "*Take a pic of you to predict your BMI*")

if file_image:
  image = Image.open(file_image)
  image = np.array(image)
  faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray_image, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
  for (x, y, w, h) in faces:
    # box bounding the face
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    bmi = predict_class(image[y:y+h, x:x+w], model)
    st.text('Your BMI value is:', bmi)
else:
    st.write("You haven't taken any picture")
