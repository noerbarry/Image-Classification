# Image Classification with Streamlit
This is a simple image classification application built using Streamlit. It allows users to upload an image and perform image classification using pretrained models from torchvision.

##Prerequisites
Before running the application, make sure you have the following dependencies installed:

Python 3.x
Streamlit
Pillow
Torch
torchvision
requests
json
You can install the required dependencies by running the following command:

pip install streamlit Pillow torch torchvision requests

##Usage
To run the application, execute the following command:

streamlit run klasifikasigambar.py

Once the application is running, you can access it in your web browser at http://localhost:8501.

Application Details
Model Selection
The application allows you to choose from the following pretrained model variants for image classification:

ResNet18
ResNet50
ResNet101
Select the desired model variant from the dropdown menu.

Image Upload
You can upload an image by clicking on the "Upload Image" button. Supported image formats are JPG, JPEG, and PNG.

Image Classification
After uploading an image, it will be displayed along with the caption "Uploaded Image". To perform image classification, click on the "Predict" button. The application will use the selected model variant to predict the class of the uploaded image. The predicted class label will be displayed below the image.

Please note that the prediction might take a few seconds, depending on the complexity of the model and the size of the image.

ImageNet Labels
The application uses the ImageNet dataset for classification. The labels for the ImageNet classes are loaded from this JSON file. These labels are used to map the predicted class index to the corresponding label.

Model Loading and Prediction
The model loading and prediction functions are implemented in the Python code. The load_model function loads a pretrained model from torchvision based on the selected model variant. The predict function takes an image and a model as input and returns the predicted class index.

Disclaimer
This application is intended for educational purposes only. The pretrained models used for image classification are based on publicly available datasets and may not be suitable for all use cases.
