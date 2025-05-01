import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

# Load scaler
scaler = joblib.load('scaler.pkl')  # Perbaiki nama file jika perlu

# Membuat dan melatih LabelEncoder untuk Geography dan Gender
geography_encoder = LabelEncoder()
gender_encoder = LabelEncoder()

# Kategori yang sudah diketahui untuk encoding
geography_encoder.fit(['France', 'Spain', 'Germany'])
gender_encoder.fit(['Male', 'Female'])

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="churn_modelling.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Judul Aplikasi
st.title("Customer Churn Prediction")
st.write("Silahkan Masukkan Data Nasabah Guna Mendapatkan Rekomendasi Untuk Status Nasabah")

# Form input pengguna
CreditScore = st.number_input("Credit Score Nasabah", min_value=0.0, max_value=1000.0, value=500.0)
Geography = st.selectbox("Geography Nasabah", ["France", "Spain", "Germany"])
Gender = st.selectbox("Jenis Kelamin Nasabah", ["Male", "Female"])
Age = st.number_input("Usia Nasabah", min_value=18, max_value=100, value=30)
Tenure = st.number_input("Lama Menjadi Nasabah (tahun)", min_value=0, max_value=10, value=3)
Balance = st.number_input("Saldo Nasabah", min_value=0.0, max_value=1000000.0, value=10000.0)
NumOfProducts = st.number_input("Jumlah Produk yang Digunakan Nasabah", min_value=1, max_value=4, value=2)
HasCrCard = st.selectbox("Apakah Nasabah Memiliki Kartu Kredit?", ["Yes", "No"])
IsActiveMember = st.selectbox("Apakah Nasabah Aktif?", ["Yes", "No"])
EstimatedSalary = st.number_input("Perkiraan Gaji Nasabah", min_value=0.0, max_value=200000.0, value=50000.0)

# Exited will be predicted by the model
if st.button("Prediksi Status Nasabah"):
    # Preprocessing input
    # Mengubah input Geography dan Gender menjadi string yang di-encode
    geography_encoded = geography_encoder.transform([Geography])[0]
    gender_encoded = gender_encoder.transform([Gender])[0]
    has_cr_card_encoded = 1 if HasCrCard == "Yes" else 0
    is_active_member_encoded = 1 if IsActiveMember == "Yes" else 0
    
    # Membuat array input untuk model
    input_data = np.array([[CreditScore, geography_encoded, gender_encoded, Age, Tenure, Balance, NumOfProducts, 
                            has_cr_card_encoded, is_active_member_encoded, EstimatedSalary]])
    input_scaled = scaler.transform(input_data).astype(np.float32)
    
    # Menjalankan model TFLite
    interpreter.set_tensor(input_details[0]['index'], input_scaled)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    
    # Menampilkan hasil prediksi
    predicted_class = int(np.argmax(prediction))
    status = "Nasabah Bertahan" if predicted_class == 0 else "Nasabah Keluar"
    
    st.success(f"Prediksi status nasabah: **{status}**")
