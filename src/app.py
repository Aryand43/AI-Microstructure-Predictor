import streamlit as st
import torch
import cv2
import numpy as np
import os
from model import MicrostructureCNN

st.set_page_config(page_title="Microstructure Predictor", layout="centered")

@st.cache_resource
def load_model():
    model = MicrostructureCNN()
    model.load_state_dict(torch.load('microstructure_model.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()
LABELS = ['d (p/h)', 'f (h/w)', 'w (width)', 'h (height)', 'p (depth)']

def predict(img_bytes):
    file_bytes = np.asarray(bytearray(img_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        return dict(zip(LABELS, output.squeeze().tolist())), img

st.title("Microstructure Predictor")

# File uploader
uploaded_file = st.file_uploader("Upload a .tif microstructure image", type=["tif", "tiff"])

if uploaded_file:
    try:
        preds, img = predict(uploaded_file.read())
        st.image(img, caption="Uploaded Microstructure", width=300, channels="GRAY")
        st.markdown("###Predicted Microstructure Properties")
        for k, v in preds.items():
            st.write(f"**{k}:** {v:.4f}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
else:
    st.info("Please upload a .tif image to get predictions.")
