import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
 
print(tf.__version__)
print(tf.config.list_physical_devices('GPU'))


# Memuat model deteksi berat sapi
model = tf.keras.models.load_model('model.h5')

# Fungsi untuk mengubah gambar menjadi array numpy
def preprocess_image(image):
    img = image.resize((90, 3))
    img = img.convert('L')  # Mengubah ke skala abu-abu (grayscale)
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    img = np.transpose(img, (0, 2, 1, 3))  # Mengubah urutan dimensi
    return img

# Fungsi untuk melakukan prediksi berat sapi
def predict_weight(image):
    processed_img = preprocess_image(image)
    prediction_in_grams = model.predict(processed_img)[0][0]
    prediction_in_kilograms = prediction_in_grams * 0.001
    return prediction_in_kilograms


# Judul aplikasi web
st.title("Deteksi Berat Sapi dengan Gambar")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar sapi", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar yang diupload
    image = Image.open(uploaded_file)
    
    # Menampilkan gambar yang diupload
    st.image(image, caption='Gambar Sapi', use_column_width=True)
    
    # Memeriksa apakah tombol "Prediksi" ditekan
    if st.button('Prediksi'):
        # Melakukan prediksi berat sapi
        weight_prediction = predict_weight(image)
        
        # Menampilkan hasil prediksi
        st.write(f"Prediksi Berat Sapi: {weight_prediction} kg")
