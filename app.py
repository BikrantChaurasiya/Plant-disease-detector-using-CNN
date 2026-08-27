import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import urllib.request

# -----------------------------
# Download model from Hugging Face
# -----------------------------

MODEL_URL = "https://huggingface.co/bikrantchaurasiya/plant_disease_model/resolve/main/plant_disease_model.keras"
MODEL_PATH = "plant_disease_model.keras"

if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


# -----------------------------
# Load model
# -----------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()


# -----------------------------
# Class names
# -----------------------------

class_names = [
    "Pepper__bell___Bacterial_spot",
    "Potato___healthy",
    "Tomato_Leaf_Mold",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Septoria_leaf_spot",
    "Tomato_healthy",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Early_blight",
    "Tomato__Target_Spot"
]


# -----------------------------
# Prediction
# -----------------------------

def predict(image):

    image = image.resize((224, 224))

    img = np.array(image)

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    index = np.argmax(prediction)

    confidence = prediction[index]

    return class_names[index], float(confidence)


# -----------------------------
# Streamlit UI
# -----------------------------

st.title("🌿 Plant Disease Detection")

st.write("Upload a plant leaf image to detect the disease.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image")

    if st.button("Predict Disease"):

        disease, confidence = predict(image)

        st.success(f"Detected Disease: {disease}")

        st.write(
            f"Confidence: {confidence * 100:.2f}%"
        )
