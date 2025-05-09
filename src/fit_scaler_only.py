import joblib
from dataset_loader import load_material_datasets
from sklearn.preprocessing import StandardScaler

dict_co, dict_steel = load_material_datasets()
full_dict = {**dict_co, **dict_steel}

sample = list(full_dict.values())[0]
label_keys = [k for k in sample.keys() if k not in ['filename', 'image_path'] and isinstance(sample[k], (int, float))]
label_data = [[row[k] for k in label_keys] for row in full_dict.values()]

scaler = StandardScaler()
scaler.fit(label_data)

joblib.dump(scaler, "label_scaler.pkl")
print("Scaler saved.")
