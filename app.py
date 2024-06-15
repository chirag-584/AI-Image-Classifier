from __future__ import division, print_function
# coding=utf-8
import os
import numpy as np
import streamlit as st
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Load your trained model
MODEL_PATH = 'modelfordashboard.h5'
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds

# Streamlit app
st.title("AI Image Classifier")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save the uploaded image
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Make prediction
    preds = model_predict(file_path, model)

    # Decode and display the result
    pred_class = "AI-generated" if preds[0][0] >= 0.5 else "Real image"
    st.write(f"Prediction: {pred_class}")

if not os.path.exists('uploads'):
    os.makedirs('uploads')

