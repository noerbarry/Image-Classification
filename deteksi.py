import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2

# Memuat model deteksi berat sapi
model_weight = tf.keras.models.load_model('model.h5')

# Fungsi untuk mengubah gambar menjadi array numpy dengan ukuran yang ditentukan
def preprocess_image(image, target_size):
    img = image.resize(target_size)
    img = img.convert('RGB')  # Mengubah ke skala warna RGB
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Fungsi untuk melakukan prediksi berat sapi
def predict_weight(image):
    processed_img = preprocess_image(image, (150, 150))
    prediction = model_weight.predict(processed_img)[0][0]
    return prediction

# Fungsi untuk konversi berat dari pounds (lbs) ke kilogram (kg)
def lbs_to_kg(weight_lbs):
    weight_kg = weight_lbs * 0.453592
    return round(weight_kg, 2)

# Fungsi untuk membaca frame dari kamera
def read_camera():
    cap = cv2.VideoCapture(0)  # Mengakses kamera dengan indeks 0
    button_counter = 0  # Initialize the button counter
    while True:
        ret, frame = cap.read()  # Membaca frame dari kamera
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Mengubah format warna BGR ke RGB
        
        # Menampilkan frame menggunakan Streamlit
        img = Image.fromarray(frame)
        st.image(img, caption='Kamera Live', use_column_width=True)
        
        button_counter += 1  # Increment the button counter
        
        # Memeriksa apakah tombol "Deteksi" ditekan
        if st.button('Deteksi' + str(button_counter), key='deteksi_button_' + str(button_counter)):
            # Melakukan prediksi berat sapi
            weight_prediction_lbs = predict_weight(img)
            
            # Konversi berat dari lbs ke kg
            weight_prediction_kg = lbs_to_kg(weight_prediction_lbs)
            
            # Menampilkan hasil prediksi
            st.write(f"Prediksi Berat Sapi: {weight_prediction_kg} kg")
        
        # Memeriksa apakah tombol "Stop" ditekan
        if st.button('Stop' + str(button_counter), key='stop_button_' + str(button_counter)):
            break

    cap.release()  # Melepaskan kamera
    cv2.destroyAllWindows()  # Menutup semua jendela OpenCV

# Judul aplikasi web
st.title("Deteksi Berat dan Klasifikasi Gambar Sapi")
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
    )

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
    
    button_counter = 0  # Initialize the button counter for the uploaded image section
    
    # Memeriksa apakah tombol "Deteksi" ditekan
    if st.button('Deteksi' + str(button_counter), key='deteksi_button_' + str(button_counter)):
        # Melakukan prediksi berat sapi
        weight_prediction_lbs = predict_weight(image)
        
        # Konversi berat dari lbs ke kg
        weight_prediction_kg = lbs_to_kg(weight_prediction_lbs)
        
        # Menampilkan hasil prediksi
        st.write(f"Prediksi Berat Sapi: {weight_prediction_kg} kg")
