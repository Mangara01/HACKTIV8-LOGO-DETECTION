import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

model = tf.keras.models.load_model("deployment/logo_model.h5")

def run():

   st.title("Logo Classifier")
   st.write("Upload an image to classify whether it's a genuine or fake logo.")

   uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

   if uploaded_file is not None:
   
      img = tf.keras.utils.load_img(uploaded_file, target_size=(70, 70))
      st.image(img, caption="Uploaded Image", use_column_width=True)

      x = tf.keras.utils.img_to_array(img) 
      x = np.expand_dims(x, axis=0) 

      images = np.vstack([x])
      classes = model.predict(images)
      result_pred = tf.where(classes < 0.7, 0, 1)

      if result_pred[0][0] == 1:
         st.write("Prediction: Real logo")
      else:
         st.write("Prediction: Fake logo")
         
if __name__ == '__main__':
    run()