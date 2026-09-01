# Satellite Land Use Classification

## Project Overview

This project focuses on multi-class image classification of satellite images using deep learning.

The EuroSAT dataset is used to classify satellite images into different land-use and land-cover categories.

Two deep learning approaches are implemented and compared:

- Custom Convolutional Neural Network (CNN)
- VGG16 model

## Dataset

The project uses the EuroSAT dataset, which contains satellite images belonging to multiple land-use and land-cover classes.

The dataset is divided into:

- Training data
- Validation data
- Testing data

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Google Colab

## Models Used

### 1. Custom CNN

A Convolutional Neural Network is built using convolution, pooling, flattening, and dense layers to classify the satellite images.

### 2. VGG16

VGG16 is used as a second deep learning approach for image classification.

## Project Workflow

1. Load the EuroSAT dataset
2. Prepare and preprocess the images
3. Create training, validation, and testing datasets
4. Build the custom CNN model
5. Train the CNN model
6. Evaluate the model
7. Build and train the VGG16 model
8. Evaluate the VGG16 model
9. Compare the model results

## Project Files

- `satellite_land_use_classification.ipynb` — Main Jupyter/Google Colab notebook containing the complete project.

## How to Run

The project was developed using Google Colab.

1. Open the notebook in Google Colab.
2. Download or provide the EuroSAT dataset.
3. Run the notebook cells sequentially.
4. Train and evaluate the models.
