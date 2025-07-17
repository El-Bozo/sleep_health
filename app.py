import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model dan tools
with open('model_sleep.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

with open('columns.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

st.title("🛌 Prediksi Gangguan Tidur")

# Input pengguna
gender = st.selectbox("Gender", label_encoders['Gender'].classes_)
age = st.slider("Age", 18, 100)
occupation = st.selectbox("Occupation", label_encoders['Occupation'].classes_)
sleep_duration = st.slider("Sleep Duration (jam)", 0.0, 12.0, 6.0)
quality_of_sleep = st.slider("Quality of Sleep", 1, 10, 5)
physical_activity = st.slider("Physical Activity Level", 0, 10, 5)
stress_level = st.slider("Stress Level", 0, 10, 5)
bmi_category = st.selectbox("BMI Category", label_encoders['BMI Category'].classes_)
heart_rate = st.slider("Heart Rate", 40, 150, 70)
daily_steps = st.number_input("Daily Steps", 0, 30000, 5000)

if st.button("Prediksi"):
    # Encode input
    row = [
        label_encoders['Gender'].transform([gender])[0],
        age,
        label_encoders['Occupation'].transform([occupation])[0],
        sleep_duration,
        quality_of_sleep,
        physical_activity,
        stress_level,
        label_encoders['BMI Category'].transform([bmi_category])[0],
        heart_rate,
        daily_steps
    ]

    # Buat DataFrame dan scaling
    input_df = pd.DataFrame([row], columns=feature_columns)
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    label = label_encoders['Sleep Disorder'].inverse_transform([prediction])[0]

    st.success(f"✅ Prediksi Gangguan Tidur: {label}")