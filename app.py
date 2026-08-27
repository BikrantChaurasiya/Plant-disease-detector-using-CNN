import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plant_disease_model.keras")

model = load_model()

# Load class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)


# Prediction function
def predict(image):

    image = image.resize((224, 224))
    image = np.array(image)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)[0]

    index = np.argmax(prediction)

    disease = class_names[index]
    confidence = prediction[index] * 100

    return disease, confidence


# UI
st.title("🌿 Plant Disease Detection")

st.write(
    "Upload a plant leaf image to detect the disease "
    "using a CNN-based deep learning model."
)

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Leaf Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Disease"):

        disease, confidence = predict(image)

        st.success(f"Detected Disease: {disease}")

        st.info(f"Confidence: {confidence:.2f}%")
