import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from model import get_resnet18, predict_from_image
from dataset_loader import load_material_datasets

# === Session Styling ===
st.set_page_config(page_title="Microstructure Analysis Dashboard", layout="wide")
st.markdown("""
<style>
    .stApp { font-family: 'Helvetica Neue', sans-serif; }
    h1, h2, h3 { color: #2c3e50; }
    .stMetricValue { font-weight: bold; color: #1abc9c; }
</style>
""", unsafe_allow_html=True)

st.title("Microstructure Analysis Dashboard")

st.sidebar.markdown("""
## Project Overview
This dashboard predicts melt pool characteristics, processing parameters, and inferred material properties from laser-melted alloy images.

Model powered by ResNet18 trained on cross-section melt pool images.
""")

label_scaler = joblib.load("label_scaler.pkl")

@st.cache_resource
def load_model_and_keys():
    dict_co, dict_steel = load_material_datasets()
    full_dict = {**dict_co, **dict_steel}
    label_keys = [
        "w", "h", "p",
        "Power", "Scanning Speed", "Powder Flow Rate",
        "Density", "Thermal Conductivity", "Specific Heat Capacity", "Thermal Expansion Coefficient"
    ]
    model = get_resnet18(output_size=len(label_keys))
    model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model, label_keys

def predict_from_tif(image_path, model, label_keys):
    try:
        preds = predict_from_image(image_path, model, label_keys, label_scaler)
        w, h, p = preds.get("w"), preds.get("h"), preds.get("p")
        preds["d"] = p / h if h else None
        preds["f"] = h / w if w else None
        return preds
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload Melt Pool Image (.tif, .jpg, .png)", type=["tif", "jpg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])
    with col1:
        img = Image.open(uploaded_file).convert("L")
        st.image(img, caption="Uploaded Image", use_column_width=True)
        img.save("temp_image.tif")

    with col2:
        model, label_keys = load_model_and_keys()
        predictions = predict_from_tif("temp_image.tif", model, label_keys)

        if predictions:
            st.session_state.history.append(predictions)

            st.success(f"""
### Prediction Summary
- **Width (w):** {predictions['w']:.2f} µm  
- **Height (h):** {predictions['h']:.2f} µm  
- **Depth (p):** {predictions['p']:.2f} µm  
""")

            st.markdown("---")
            st.subheader("Predicted Melt Pool Geometry")
            fig, ax = plt.subplots()
            ax.bar(["Width", "Height", "Depth"], [predictions['w'], predictions['h'], predictions['p']], color=['#3498db', '#2ecc71', '#e74c3c'])
            ax.set_ylabel("µm")
            ax.set_title("Predicted Melt Pool Geometry")
            st.pyplot(fig)

            st.markdown("---")
            st.subheader("Predicted Process Parameters")
            st.metric("Power", f"{predictions['Power']:.2f} W")
            st.metric("Scanning Speed", f"{predictions['Scanning Speed']:.2f} mm/min")
            st.metric("Powder Flow Rate", f"{predictions['Powder Flow Rate']:.2f} g/min")

            st.markdown("---")
            st.subheader("Derived Metrics")
            st.metric("Dilution Ratio (d = p/h)", f"{predictions['d']:.4f}" if predictions['d'] else "N/A")
            st.metric("Form Factor (f = h/w)", f"{predictions['f']:.4f}" if predictions['f'] else "N/A")

            st.markdown("---")
            st.subheader("Predicted Material Properties")
            st.metric("Density", f"{predictions['Density']:.2f} kg/m³")
            st.metric("Thermal Conductivity", f"{predictions['Thermal Conductivity']:.2f} W/m·K")
            st.metric("Specific Heat Capacity", f"{predictions['Specific Heat Capacity']:.2f} J/kg·K")
            st.metric("Thermal Expansion Coefficient", f"{predictions['Thermal Expansion Coefficient']:.6f} 1/K")

            st.markdown("---")
            csv = pd.DataFrame([predictions]).to_csv(index=False).encode("utf-8")
            st.download_button("Download Predictions (.csv)", csv, "predictions.csv", "text/csv")

            st.markdown("---")
            st.markdown("### Session Prediction History")
            st.dataframe(pd.DataFrame(st.session_state.history).round(2))

st.markdown("""
---
**Footer**  
Model powered by ResNet18 – trained on cross-section melt pool images.  
© 2025 SC3DP. All rights reserved.
""")