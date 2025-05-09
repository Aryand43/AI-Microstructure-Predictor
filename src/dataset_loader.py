import os
import pandas as pd
import torch

def load_material_datasets(
    images_dir='../data/images/',
    images2_dir='../data/images2/',
    images3_dir='../data/images3/',
    labels1_csv='../data/labels.csv',
    labels2_csv='../data/labels2.csv',
    labels3_csv='../data/labels3.csv'
):
    co_properties = {
        "density": 8220,
        "thermal_conductivity": 12.89,
        "specific_heat": 5110.37,
        "thermal_expansion": 1.42e-5
    }

    steel_properties = {
        "density": 7660,
        "thermal_conductivity": 18.31,
        "specific_heat": 453.13,
        "thermal_expansion": 2.30e-5
    }

    h13_properties = {
        "density": 7800,
        "thermal_conductivity": 24.3,
        "specific_heat": 460,
        "thermal_expansion": 1.2e-5
    }

    dict_co = create_material_dict(images_dir, labels1_csv, co_properties)
    dict_steel = create_material_dict(images2_dir, labels2_csv, steel_properties)
    dict_h13 = create_material_dict(images3_dir, labels3_csv, h13_properties)

    print("INSIDE DATASET LOADER:")
    print("dict_co type:", type(dict_co))
    print("dict_steel type:", type(dict_steel))
    print("dict_h13 type:", type(dict_h13))

    return dict_co, dict_steel, dict_h13


KEY_REMAP = {
    'Power (W)': 'Power',
    'Scanning speed (mm/min)': 'Scanning Speed',
    'Powder flow rate (g/min)': 'Powder Flow Rate',
    'Power': 'Power',
    'Scanning speed': 'Scanning Speed',
    'Powder flow rate': 'Powder Flow Rate',
    'density': 'Density',
    'thermal_conductivity': 'Thermal Conductivity',
    'specific_heat': 'Specific Heat Capacity',
    'thermal_expansion': 'Thermal Expansion Coefficient'
}

def remap_keys(entry):
    return {KEY_REMAP.get(k.lower().strip(), k): v for k, v in entry.items()}

def safe_numeric_key(filename):
    name_part = os.path.splitext(filename)[0]
    return int(name_part) if name_part.isdigit() else float('inf')

def create_material_dict(images_dir, labels_csv, material_properties):
    df = pd.read_csv(labels_csv)
    df.columns = df.columns.str.strip().str.lower()
    input_candidates = [
        ["power", "scanning speed", "powder flow rate"],
        ["power (w)", "scanning speed (mm/min)", "powder flow rate (g/min)"]
    ]
    required_inputs = next((c for c in input_candidates if all(k in df.columns for k in c)), None)
    if required_inputs is None:
        raise ValueError(f"[ERROR] No matching input column set found in {labels_csv}. Available columns: {df.columns.tolist()}")

    df = df.dropna(subset=required_inputs).reset_index(drop=True)

    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith('.tif')],
        key=safe_numeric_key
    )

    if len(image_files) != len(df):
        print(f"[WARNING] Image-label mismatch: {len(image_files)} images vs {len(df)} labels")

    material_dict = {}
    for idx, filename in enumerate(image_files):
        if idx < len(df):
            row = df.iloc[idx].to_dict()
            row = remap_keys(row)
            row.update(remap_keys(material_properties))
            row["image_path"] = os.path.join(images_dir, filename)
            material_dict[filename] = row

    return material_dict


if __name__ == "__main__":
    co_data, steel_data, h13_data = load_material_datasets()
    print(f"[CoCrFeNi] Loaded {len(co_data)} entries")
    print(f"[Steel] Loaded {len(steel_data)} entries")
    print(f"[H13] Loaded {len(h13_data)} entries")
