

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib, random, os
from pathlib import Path

_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
_COLS = ["id", "clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
         "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei",
         "bland_chromatin", "normal_nucleoli", "mitoses", "class"]
_FEATURE_COLS = _COLS[1:-1]

def load_data_set():
    df = pd.read_csv(_URL, names=_COLS)
    df = df.replace("?", np.nan).dropna()
    df[_FEATURE_COLS] = df[_FEATURE_COLS].astype(float)
    df["label"] = df["class"].map({2: 0, 4: 1}).astype(int)
    
    X = df[_FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    return X, y

X, y = load_data_set()
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape)



from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib
import torch
import os

class TestLoaderWBC:
    """
    Interface for the Wisconsin Breast Cancer dataset.
    Provides deterministic DataLoader access and per-sample retrieval.
    """
    def __init__(self, batch_size=64, test_size=0.2, seed=42,
                 scaler_path="artifacts/breast_minmax_scaler.joblib",
                 shuffle=True):
       
        X, y = load_data_set()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y, shuffle=True
        )

        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        else:
            scaler = MinMaxScaler().fit(X_train)
            os.makedirs(os.path.dirname(scaler_path) or ".", exist_ok=True)
            joblib.dump(scaler, scaler_path)

        X_test_sc = scaler.transform(X_test).astype("float32")

        self.dataset = TensorDataset(torch.from_numpy(X_test_sc),
                                     torch.from_numpy(y_test).long())

        g = torch.Generator()
        g.manual_seed(seed)

        self.loader = DataLoader(self.dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 generator=g)

    def get_loader(self):
        return self.loader

    def get_test_sample(self, i: int):
        sample,label=self.dataset[i]

        return sample,label.item()
    


testloader = TestLoaderWBC(batch_size=128, seed=123)


for xb, yb in testloader.get_loader():
    print("Labels batch:", yb[:10])
    break

test_sample= testloader.get_test_sample(42)
print("test sample label:", test_sample)