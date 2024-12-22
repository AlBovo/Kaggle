import numpy as np
from tqdm import tqdm
from npy_append_array import NpyAppendArray
from data.utils import load_csv, parse_row

def generate_data(file_path):
    header, data = load_csv(file_path)
    with (NpyAppendArray("features.npy", delete_if_exists=True) as features,
          NpyAppendArray("results.npy", delete_if_exists=True) as results):
        for row in tqdm(data, desc="Generating data"):
            casted = parse_row(row)
            features.append(np.array([casted]))
            results.append(np.array([float(row[-1])]))
    print("Data generated successfully.")

def generate_test(file_path):
    header, data = load_csv(file_path)
    with NpyAppendArray("test_f.npy", delete_if_exists=True) as test_f:
        for row in tqdm(data, desc="Generating test data"):
            casted = parse_row(row)
            test_f.append(np.array([casted]))
    print("Test data generated successfully.")