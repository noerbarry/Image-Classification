import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Memuat model deteksi berat sapi
model_weight = tf.keras.models.load_model('model.h5')

# Memuat model klasifikasi sapi atau bukan
model_classification = tf.keras.models.load_model('classification_model.h5')

# Fungsi untuk mengubah gambar menjadi array numpy
def preprocess_image(image):
    img = image.resize((150, 150))
    img = img.convert('RGB')  # Mengubah ke skala warna RGB
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Fungsi untuk melakukan prediksi berat sapi
def predict_weight(image):
    processed_img = preprocess_image(image)
    prediction = model_weight.predict(processed_img)[0][0]
    return prediction

# Fungsi untuk melakukan prediksi klasifikasi sapi atau bukan
def predict_classification(image):
    processed_img = preprocess_image(image)
    prediction = model_classification.predict(processed_img)
    if prediction > 0.5:
        return "Sapi"
    else:
        return "Bukan Sapi"

# Fungsi untuk membaca frame dari kamera
def read_camera():
    cap = cv2.VideoCapture(0)  # Mengakses kamera dengan indeks 0
    while True:
        ret, frame = cap.read()  # Membaca frame dari kamera
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mengubah format warna BGR ke RGB
        
        # Menampilkan frame menggunakan Streamlit
        img = Image.fromarray(frame)
        st.image(img, caption='Kamera Live', use_column_width=True)
        
        # Memeriksa apakah tombol "Deteksi" ditekan
        if st.button('Deteksi'):
            # Melakukan prediksi berat sapi
            weight_prediction = predict_weight(img)
            
            # Melakukan prediksi klasifikasi sapi atau bukan
            class_prediction = predict_classification(img)
            
            # Menampilkan hasil prediksi
            st.write(f"Prediksi Berat Sapi: {weight_prediction} kg")
            st.write(f"Prediksi Klasifikasi: {class_prediction}")
            
        # Menghentikan streaming video saat tombol 'Stop' ditekan
        if st.button('Stop'):
            break

    cap.release()  # Melepaskan kamera
    cv2.destroyAllWindows()  # Menutup semua jendela OpenCV

# Judul aplikasi web
st.title("Deteksi Berat dan Klasifikasi Gambar Sapi")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar", type=["jpg", "jpeg", "png"])

# Memeriksa apakah tombol 'Start' ditekan
if st.button('Start Kamera'):
    # Memanggil fungsi untuk membaca kamera
    read_camera()

if uploaded_file is not None:
    # Membaca gambar yang diupload
    image = Image.open(uploaded_file)
    
    # Menampilkan gambar yang diupload
    st.image(image, caption='Gambar', use_column_width=True)
    
    # Memeriksa apakah tombol "Deteksi" ditekan
    if st.button('Deteksi'):
        # Melakukan prediksi berat sapi
        weight_prediction = predict_weight(image)
        
        # Melakukan prediksi klasifikasi sapi atau bukan
        class_prediction = predict_classification(image)
        
        # Menampilkan hasil prediksi
        st.write(f"Prediksi Berat Sapi: {weight_prediction} kg")
        st.write(f"Prediksi Klasifikasi: {class_prediction}")
