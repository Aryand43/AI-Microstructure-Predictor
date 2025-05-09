# === Microstructure Analysis Dashboard v2 (SC3DP Enhanced) ===
import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from model import get_resnet
from dataset_loader import load_material_datasets

# === Streamlit UI Setup ===
st.set_page_config(page_title="Microstructure Analysis Dashboard", layout="wide")
st.markdown("""
<style>
    .stApp { font-family: 'Helvetica Neue', sans-serif; }
    h1, h2, h3 { color: #0b3d91; }
    .stMetricValue { font-weight: bold; color: #1abc9c; }
    .material-header { background-color: #e8f5e9; padding: 5px 10px; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# === Title & Version Info ===
st.title("Microstructure Analysis Dashboard")
st.caption("Model v1.2 | Dataset v2025-05 | Powered by ResNet34 | SC3DP Internal")

# === Load Model & Label Keys ===
label_scaler = joblib.load("label_scaler.pkl")

@st.cache_resource
def load_model_and_keys():
    dict_co, dict_steel, dict_h13 = load_material_datasets()
    full_dict = {"CoCrFeNi": dict_co, "Steel": dict_steel, "H13": dict_h13}
    label_keys = [
        "w", "h", "p",
        "Power", "Scanning Speed", "Powder Flow Rate",
        "Density", "Thermal Conductivity", "Specific Heat Capacity", "Thermal Expansion Coefficient"
    ]
    model = get_resnet(model_name="resnet34", output_size=len(label_keys))
    model.load_state_dict(torch.load("best_model.pt", map_location=torch.device("cpu")))
    model.eval()
    return model, label_keys, full_dict

# === Prediction Function ===
def predict_from_tif(image_path, model, label_keys):
    try:
        from model import predict_from_image
        preds = predict_from_image(image_path, model, label_keys, label_scaler)
        w, h, p = preds.get("w"), preds.get("h"), preds.get("p")
        preds["d"] = p / h if h else None
        preds["f"] = h / w if w else None
        return preds
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# === Sidebar: Session Summary Panel ===
st.sidebar.title("Session Summary")
material_choice = st.sidebar.selectbox("Select Material Type", ["CoCrFeNi", "Steel", "H13"])
advanced_mode = st.sidebar.checkbox("Enable Advanced Mode")

if "history" not in st.session_state:
    st.session_state.history = []

# === Upload + Batch Support ===
st.header("Upload Melt Pool Image(s)")
uploaded_files = st.file_uploader("Upload one or more images", type=["tif", "jpg", "png"], accept_multiple_files=True)

if uploaded_files:
    model, label_keys, full_dict = load_model_and_keys()
    selected_dict = full_dict[material_choice]

    for uploaded_file in uploaded_files:
        st.markdown(f"### Analyzing: `{uploaded_file.name}`")
        cols = st.columns([1, 2])
        img = Image.open(uploaded_file).convert("L")
        cols[0].image(img, caption="Uploaded Image", use_column_width=True)

        # Save temporarily
        temp_path = f"temp_{uploaded_file.name}"
        img.save(temp_path)

        preds = predict_from_tif(temp_path, model, label_keys)

        if preds:
            st.session_state.history.append(preds)
            outliers = []
            if preds["Power"] > np.mean([x["Power"] for x in st.session_state.history]) + 2 * np.std([x["Power"] for x in st.session_state.history]):
                outliers.append("Power is unusually high")

            # === TABS ===
            tabs = st.tabs(["Geometry", "Process Parameters", "Material Properties", "CSV Export", "History"])

            with tabs[0]:
                st.metric("Width (w)", f"{preds['w']:.2f} µm", help="Melt pool width in microns")
                st.metric("Height (h)", f"{preds['h']:.2f} µm")
                st.metric("Depth (p)", f"{preds['p']:.2f} µm")
                st.metric("Dilution Ratio (p/h)", f"{preds['d']:.4f}" if preds['d'] else "N/A")
                st.metric("Form Factor (h/w)", f"{preds['f']:.4f}" if preds['f'] else "N/A")
                fig, ax = plt.subplots()
                ax.bar(["Width", "Height", "Depth"], [preds['w'], preds['h'], preds['p']], color=["#1f77b4", "#2ca02c", "#d62728"])
                ax.set_ylabel("µm")
                ax.set_title("Melt Pool Geometry")
                st.pyplot(fig)

            with tabs[1]:
                st.metric("Power", f"{preds['Power']:.2f} W")
                st.metric("Scanning Speed", f"{preds['Scanning Speed']:.2f} mm/min")
                st.metric("Powder Flow Rate", f"{preds['Powder Flow Rate']:.2f} g/min")
                if advanced_mode:
                    st.warning("Advanced Engineering Note: Watch for low flow rate + high power combinations.")
                if outliers:
                    st.error("\n".join(outliers))

            with tabs[2]:
                st.metric("Density", f"{preds['Density']:.2f} kg/m³")
                st.metric("Thermal Conductivity", f"{preds['Thermal Conductivity']:.2f} W/m·K")
                st.metric("Specific Heat Capacity", f"{preds['Specific Heat Capacity']:.2f} J/kg·K")
                st.metric("Thermal Expansion Coefficient", f"{preds['Thermal Expansion Coefficient']:.6f} 1/K")

            with tabs[3]:
                csv = pd.DataFrame([preds]).to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, f"{uploaded_file.name}_prediction.csv", "text/csv")

            with tabs[4]:
                st.dataframe(pd.DataFrame(st.session_state.history).round(2))
                if st.button("Clear History"):
                    st.session_state.history.clear()
                    st.experimental_rerun()

# === Footer ===
st.markdown("""
---
<center>
<i>SC3DP | Nanyang Technological University | Microstructure Predictor – ResNet34 (v1.2)</i>
</center>
""")