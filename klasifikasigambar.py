import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import requests
import json

# Mengunduh file label ImageNet
label_url = 'https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json'
response = requests.get(label_url)
labels = json.loads(response.content.decode())

# Fungsi untuk memuat model klasifikasi gambar
def load_model(model_name):
    model_func = getattr(models, model_name)
    model = model_func(pretrained=True)
    model.eval()
    return model

# Fungsi untuk melakukan prediksi kelas gambar
def predict(image, model):
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to('cuda') if torch.cuda.is_available() else input_batch
    model = model.to('cuda') if torch.cuda.is_available() else model
    with torch.no_grad():
        output = model(input_batch)
    _, predicted_idx = torch.max(output, 1)
    return predicted_idx.item()

# Memuat model saat aplikasi dimulai
model_variant = st.selectbox("Pilih varian model", ['resnet18', 'resnet50', 'resnet101'])
model = load_model(model_variant)

# Mengatur judul aplikasi
st.title("Klasifikasi Gambar")

# Mengunggah gambar dari pengguna
uploaded_image = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

# Jika gambar telah diunggah, tampilkan dan lakukan prediksi
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    if st.button('Prediksi'):
        predicted_class = predict(image, model)
        predicted_label = labels[predicted_class]
        st.write(f"Kelas prediksi: {predicted_label}")
