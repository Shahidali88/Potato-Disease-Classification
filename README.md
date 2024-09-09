# Potato-Disease-Classification
CNN-based Potato Disease Classification using 
Potato Disease Classification Using CNN
This project demonstrates how to classify potato plant leaves into three categories: Potato Early Blight, Potato Late Blight, and Healthy using a Convolutional Neural Network (CNN) implemented in TensorFlow. The model is trained on the PlantVillage dataset, specifically on images of potato leaves.

Table of Contents
Project Overview
Dataset
Model Architecture
Installation
Usage
Results
Evaluation
Future Work
License
Project Overview
The main goal of this project is to build a robust classification model that can detect early and late blight in potato leaves and distinguish them from healthy leaves. The project involves:

Data preprocessing and augmentation
Building and training a CNN model
Evaluating the model's performance using metrics such as accuracy, precision, recall, and F1-score
Saving the model for future use
Dataset
The dataset used in this project is sourced from the PlantVillage dataset. It contains labeled images of potato plant leaves in three categories:

Potato Early Blight
Potato Late Blight
Healthy
The dataset was downloaded from PlantVillage on Kaggle.

Model Architecture
The CNN model used for classification is built using the TensorFlow framework. It consists of multiple layers:

Convolutional Layers: Extracting features from images.
Pooling Layers: Reducing the spatial dimensions.
Dense Layers: Fully connected layers for classification.
Data Augmentation: Random flip, rotation, and zoom to make the model robust.
Key Layers:

6 Convolutional layers (32, 64, and 128 filters)
MaxPooling layers
Dropout (to prevent overfitting)
Output layer with softmax activation for multi-class classification.
Installation
To run this project, ensure you have the following dependencies installed:

bash
Copy code
pip install tensorflow
pip install numpy
pip install matplotlib
pip install kaggle
