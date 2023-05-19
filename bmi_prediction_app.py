filename = '/home/appuser/venv/lib/python3.9/site-packages/keras_vggface/models.py'
text = open(filename).read()
open(filename, 'w+').write(text.replace('keras.engine.topology', 'tensorflow.keras.utils'))

import streamlit as st
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp
import keras.utils as image
from keras_vggface.utils import preprocess_input
import numpy as np
import time
import io
import pandas as pd

def pearson_corr(y_test, y_pred):
  corr = tfp.stats.correlation(y_test, y_pred)
  return corr

# def custom_object_scope(custom_objects):
#   return tf.keras.utils.CustomObjectScope(custom_objects)

# with custom_object_scope({'pearson_corr': pearson_corr}):
#   model = load_model('My_model_vgg16.h5')

model = load_model('My_model_vgg16.h5', compile=False)

def predict_class(image, model):
  img = image.copy()
  img = cv2.resize(img, (224, 224))
  img = np.array(img).astype(np.float32)
  img = np.expand_dims(img, axis = 0)
  img = preprocess_input(img, version=2)
  prediction = model.predict(img)[0][0]
  return prediction

def process_img(file_image):
  image = Image.open(file_image)
  image = np.array(image)
  faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faces = faceCascade.detectMultiScale(gray_image, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
  if len(faces) == 0:
    st.warning('No face detected! Please take it again.')
  for (x, y, w, h) in faces:
    # box bounding the face
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    bmi = predict_class(image[y:y+h, x:x+w], model)
    cv2.putText(image, f'BMI:{bmi}', (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
  pred_image = image
  return pred_image

def calculator(height, weight):
  score = 730 * weight / height**2
  col3.success(f'Your BMI value is: {score}')
  
def change_photo_state():
  st.session_state['photo'] = 'Done'

def main():
  if 'photo' not in st.session_state:
    st.session_state['photo'] = 'Not done'

  st.set_page_config(layout="centered", page_icon='random', )
  st.markdown("""
  <style>
  .big-font {
      font-size:90px !important;
  }
  </style>
  """, unsafe_allow_html=True)

  st.markdown('<p class="big-font">BMI Prediction 📸</p>', unsafe_allow_html=True)
  bmi_img = Image.open('bmi.jpeg')
  st.image(bmi_img)
  #st.title('*BMI prediction 📸*')
  st.subheader('Body Mass Index(BMI) estimates the total body fat and assesses the risks for diseases related to increase body fat. A higher BMI may indicate higher risk of developing many diseases.')
  st.write('* Since we only have the access to your face feature, the estimated value is biased')
  col2, col3 = st.columns([2,1])

  upload_img = col3.file_uploader('Upload a photo 🖼', on_change=change_photo_state)
  file_image = col2.camera_input('Take a pic of you 😊', on_change=change_photo_state)
  index = {'BMI':['16<BMI<=18.5', '18.5<BMI<=25', '25<BMI<=30', '30<BMI<=35', '35<BMI<=40', '40<BMI'],
           'WEIGHT STATUS':['Underweight', 'Normal', 'Overweight', 'Moderately obese', 'Severely obese', 'Very severely obese']}
  df = pd.DataFrame(data=index)
  col3.table(df)
  expander = col3.expander('BMI Index')
  expander.write('The table above shows the standard weight status categories based on BMI for people ages 20 and older. (Note: This is just the reference, please consult professionals for more health issues.)')           
  
  feet = col3.number_input(label='Height(feet)')
  inch = col3.number_input(label='Height(inches)')
  weight = col3.number_input(label='Weight(pounds)')
  
  if col3.button('Calculate BMI'):
    height = feet * 12 + inch
    calculator(height, weight)

  if st.session_state['photo'] == 'Done':
    process_bar3 = col3.progress(0, text='🏃‍♀️')

    process_bar2 = col2.progress(0, text='🏃')

    if file_image:
      for process in range(100):
        time.sleep(0.01)
        process_bar2.progress(process+1)
      col2.success('Taken the photo sucessfully!')
      pred_camera = process_img(file_image)
      st.image(pred_camera, caption='Predicted photo')
      image = Image.fromarray(pred_camera)
      # Convert the PIL Image to bytes
      image_bytes = io.BytesIO()
      image.save(image_bytes, format='PNG')
      image_bytes = image_bytes.getvalue()
      col3.divider()
      col3.write('Download the predicted image if you want!')
      download_img = col3.download_button(
        label=':black[Download image]', 
        data=image_bytes,
        file_name=file_image.name.split('.')[0] + '_bmi.png',
        mime="image/png",
        use_container_width=True)
    elif upload_img:
      for process in range(100):
        time.sleep(0.01)
        process_bar3.progress(process+1)
      col3.success('Uploaded the photo sucessfully!')
      pred_upload = process_img(upload_img)
      st.image(pred_upload, caption='Predicted photo')
      image = Image.fromarray(pred_upload)
      # Convert the PIL Image to bytes
      image_bytes = io.BytesIO()
      image.save(image_bytes, format='PNG')
      image_bytes = image_bytes.getvalue()
      col3.divider()
      col3.write('Download the predicted image if you want!')
      download_img = col3.download_button(
        label='Download image', 
        data=image_bytes,
        file_name=upload_img.name.split('.')[0]  + '_bmi.png',
        mime="image/png")
 
if __name__=='__main__':
    main()
