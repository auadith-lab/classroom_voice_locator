import os
import numpy as np
import pandas as pd
import librosa

# ===== SETTINGS =====
DATASET_PATH = "dataset"
OUTPUT_CSV = "features.csv"
SAMPLE_RATE = 44100

# ===== FUNCTION TO EXTRACT FEATURES =====
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    features = []

    # 1️⃣ MFCC (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    features.extend(mfcc_mean)

    # 2️⃣ Energy (RMS)
    energy = np.mean(librosa.feature.rms(y=y))
    features.append(energy)

    # 3️⃣ Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    features.append(zcr)

    # 4️⃣ Spectral Centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features.append(centroid)

    return features

# ===== PROCESS DATASET =====
data = []
labels = ["teacher", "students", "noise"]

for label in labels:
    folder_path = os.path.join(DATASET_PATH, label)

    for file in os.listdir(folder_path):
        if file.endswith(".wav"):
            file_path = os.path.join(folder_path, file)

            try:
                features = extract_features(file_path)
                features.append(label)
                data.append(features)
                print("Processed:", file)

            except Exception as e:
                print("Error processing", file, ":", e)

# ===== CREATE CSV =====
columns = []

for i in range(13):
    columns.append(f"MFCC_{i+1}")

columns.extend(["Energy", "ZCR", "Centroid", "Label"])

df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)

print("\n✅ Feature extraction completed!")
print("Saved as:", OUTPUT_CSV)