
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(page_title="White Spot Classifier", layout="centered")

st.title("🦐 White Spot Disease Classifier")
st.write("Upload or capture an image to classify shrimp as **Healthy** or **Infected**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
capture_image = st.camera_input("Take a photo")

model = load_model("white_spot_cnn_model1_float16.h5")
CLASS_NAMES = ['Healthy', 'Infected']

def predict(image):
    image = image.resize((128, 128))
    img_array = img_to_array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    preds = model.predict(img_array)[0]
    return preds

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif capture_image is not None:
    image = Image.open(capture_image)

if image:
    st.image(image, caption="Uploaded Image", use_column_width=True)
    preds = predict(image)
    class_idx = np.argmax(preds)
    label = CLASS_NAMES[class_idx]
    confidence = float(preds[class_idx]) * 100

    st.markdown(f"### 🧪 Prediction: `{label}`")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

    fig, ax = plt.subplots()
    ax.pie(preds, labels=CLASS_NAMES, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    st.pyplot(fig)
