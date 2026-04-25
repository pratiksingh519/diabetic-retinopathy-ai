import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Eye Detection", page_icon="👁️", layout="centered")

# --- CUSTOM CSS (Design 🔥) ---
st.markdown("""
    <style>
    .main {
        background-color: #0f172a;
        color: white;
    }
    .stButton>button {
        background-color: #22c55e;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stFileUploader {
        border: 2px dashed #22c55e;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOAD ---
model = tf.keras.models.load_model("model.h5")
classes = ['No_DR','Mild','Moderate','Severe','Proliferative_DR']

# --- TITLE ---
st.markdown("<h1 style='text-align: center; color: #22c55e;'>🧠 AI Eye Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a retinal image and detect Diabetic Retinopathy instantly</p>", unsafe_allow_html=True)

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # --- IMAGE DISPLAY ---
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # --- PREDICT BUTTON ---
    if st.button("🔍 Analyze Image"):
        img = np.array(image)
        img = cv2.resize(img, (224,224))
        img = img / 255.0
        img = np.reshape(img, (1,224,224,3))

        pred = model.predict(img)
        result = classes[np.argmax(pred)]

        # --- RESULT DISPLAY ---
        if result == "No_DR":
            st.success(f"✅ Result: {result} (Healthy Eye)")
        elif result == "Mild":
            st.warning(f"⚠️ Result: {result}")
        else:
            st.error(f"🚨 Result: {result} (Consult Doctor)")