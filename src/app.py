import streamlit as st
import torch
import cv2
import numpy as np
from model import MicrostructureCNN

# --- Setup ---
st.set_page_config(page_title="AI-Driven Microstructure Tool", layout="centered")
st.title("AI-Driven Microstructure Tool")

# --- Tool Selection ---
tool = st.sidebar.selectbox("Select Tool", [
    "Bead Geometry Predictor",
    "Process Parameter Optimizer"
])

# --- Bead Geometry Predictor ---
if tool == "Bead Geometry Predictor":
    st.header("Bead Geometry Predictor")
    uploaded_file = st.file_uploader("Upload a .tif cross-section image", type=["tif"])

    if uploaded_file is not None:
        # Load image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0).unsqueeze(0)  # [1, 1, 224, 224]

        # Load model
        model = MicrostructureCNN()
        model.load_state_dict(torch.load("microstructure_model.pth", map_location=torch.device('cpu')))
        model.eval()

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            w, h, p = output.squeeze().tolist()
            f = h / w if w != 0 else 0
            d = p / h if h != 0 else 0

        # Display results
        st.subheader("Predicted Geometry")
        st.write(f"**Width (w):** {w:.2f} µm")
        st.write(f"**Height (h):** {h:.2f} µm")
        st.write(f"**Depth (p):** {p:.2f} µm")
        st.markdown("---")
        st.subheader("Derived Ratios")
        st.write(f"**f = h/w:** {f:.2f}")
        st.write(f"**d = p/h:** {d:.2f}")

# --- Process Parameter Optimizer ---
elif tool == "Process Parameter Optimizer":
    st.header("Process Parameter Optimizer")
    st.info("Coming soon: Input w, h, p and get predicted Laser Power, Scan Speed, and Powder Flow Rate.")