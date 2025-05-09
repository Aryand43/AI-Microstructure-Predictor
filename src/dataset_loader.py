import os
import pandas as pd
import torch

def load_material_datasets(
    images_dir='../data/images/',
    images2_dir='../data/images2/',
    labels1_csv='../data/labels.csv',
    labels2_csv='../data/labels2.csv'
):
    # Material properties
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

    # Read CSVs
    df_co = pd.read_csv(labels1_csv).dropna().reset_index(drop=True)
    df_steel = pd.read_csv(labels2_csv).dropna().reset_index(drop=True)

    # Collect image filenames
    images1_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith('.tif')])
    images2_files = sorted([f for f in os.listdir(images2_dir) if f.lower().endswith('.tif')])

    dict_co = {}
    dict_steel = {}

    for idx, filename in enumerate(images1_files):
        if idx < len(df_co):
            row = df_co.iloc[idx].to_dict()
            row = remap_keys(row)
            row.update(remap_keys(co_properties))
            row["image_path"] = os.path.join(images_dir, filename)
            dict_co[filename] = row

    for idx, filename in enumerate(images2_files):
        if idx < len(df_steel):
            row = df_steel.iloc[idx].to_dict()
            row = remap_keys(row)
            row.update(remap_keys(steel_properties))
            row["image_path"] = os.path.join(images2_dir, filename)
            dict_steel[filename] = row  

    print("INSIDE DATASET LOADER:")
    print("dict_co type:", type(dict_co))
    print("dict_steel type:", type(dict_steel))

    return dict_co, dict_steel

KEY_REMAP = {
    'Power (W)': 'Power',
    'Scanning speed (mm/min)': 'Scanning Speed',
    'Powder flow rate (g/min)': 'Powder Flow Rate',
    'density': 'Density',
    'thermal_conductivity': 'Thermal Conductivity',
    'specific_heat': 'Specific Heat Capacity',
    'thermal_expansion': 'Thermal Expansion Coefficient'
}

def remap_keys(entry):
    return {KEY_REMAP.get(k, k): v for k, v in entry.items()}


if __name__ == "__main__":
    co_data, steel_data = load_material_datasets()
    print(f"[CoCrFeNi] Loaded {len(co_data)} entries")
    print(f"[H13Steel] Loaded {len(steel_data)} entries")
    print(type(co_data))
    print(type(steel_data))
