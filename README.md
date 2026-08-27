# 🌿 Plant Disease Detection Using CNN

A deep learning-based image classification project for detecting plant diseases from leaf images using a Convolutional Neural Network (CNN).

The system takes a plant leaf image as input and predicts the corresponding disease class with a confidence score.

## 🚀 Live Demo

👉 **[Try the Plant Disease Detection App](https://plant-disease-detector-using-cnn-zfoqggcm5yfzqzvrpobsnh.streamlit.app/)**

---

## 📌 Project Overview

Plant diseases can significantly affect crop production and quality. Early identification of diseases can help farmers take appropriate action.

This project uses a custom CNN model trained on the PlantVillage dataset to automatically classify plant leaf images into different disease categories.

### Key Features

- 🌱 Plant leaf disease classification
- 🧠 Custom CNN architecture using TensorFlow/Keras
- 🖼️ 224 × 224 image input
- 🔍 15 disease/healthy classes
- 📊 Accuracy, Precision, Recall and F1-Score evaluation
- 📈 Training and validation accuracy/loss visualization
- 🔲 Confusion matrix
- 📋 Classification report
- 🌐 Web-based prediction interface using Streamlit

---

## 🧠 CNN Architecture

The model consists of:

- Rescaling layer
- 3 Convolutional layers
- 3 MaxPooling layers
- Flatten layer
- 2 Dropout layers
- 2 Fully Connected (Dense) layers
- Softmax output layer

### Architecture

```text
Input Image (224 × 224 × 3)
            ↓
      Rescaling (1/255)
            ↓
     Conv2D - 32 filters
            ↓
       MaxPooling
            ↓
     Conv2D - 64 filters
            ↓
       MaxPooling
            ↓
     Conv2D - 128 filters
            ↓
       MaxPooling
            ↓
          Flatten
            ↓
       Dropout (0.3)
            ↓
        Dense (100)
            ↓
       Dropout (0.3)
            ↓
         Dense (50)
            ↓
      Dense (15, Softmax)
            ↓
      Disease Prediction
