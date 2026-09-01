import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('best_model.keras')
class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
               'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

st.title("Satellite Land-Use Classifier")
uploaded_file = st.file_uploader("Upload a satellite image", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((64, 64))
    st.image(img, caption="Uploaded Image")
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")