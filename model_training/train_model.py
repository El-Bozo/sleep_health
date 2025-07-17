import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')

# Drop kolom tidak perlu dan rows yang tidak berlabel
df = df.drop(columns=['Person ID', 'Blood Pressure'])
df = df[df['Sleep Disorder'].notna()].copy()

# Encode kategorikal
label_encoders = {}
for col in ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Simpan nama kolom fitur
feature_columns = df.drop(columns=['Sleep Disorder']).columns.tolist()

# Pisah fitur dan target
X = df[feature_columns]
y = df['Sleep Disorder']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model dan scaler
output_dir = '../sleep_dashboard'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'model_sleep.pkl'), 'wb') as f:
    pickle.dump(model, f)

with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

with open(os.path.join(output_dir, 'label_encoders.pkl'), 'wb') as f:
    pickle.dump(label_encoders, f)

with open(os.path.join(output_dir, 'columns.pkl'), 'wb') as f:
    pickle.dump(feature_columns, f)

print("✅ Model dan dependensi berhasil disimpan!")