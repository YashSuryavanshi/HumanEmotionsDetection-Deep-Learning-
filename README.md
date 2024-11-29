# Human Emotions Detection Using Deep Learning

This project focuses on building a system to classify emotions from facial images using deep learning techniques. The project incorporates Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) to achieve high accuracy in emotion classification.

## Project Overview
 
- **Objective:** To develop a robust system for emotion classification with improved accuracy compared to baseline models.  
- **Dataset:** [Human Emotions Dataset](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes) from Kaggle.  

## Features

- **Deep Learning Models:** 
  - Implemented CNNs and Vision Transformers for image-based emotion classification.
  - Optimized models using TensorFlow and Keras callbacks to achieve an accuracy improvement from 71.1% to 90%.
  
- **Transfer Learning:** 
  - Integrated pre-trained models to enhance classification performance.  

- **Model Interpretability:** 
  - Utilized Grad-CAM to visualize and interpret model decisions, providing insights into the learned features.  

- **Deployment:** 
  - Converted trained models to ONNX format with quantization for efficient deployment.  
  - Built APIs for the system using FastAPI and conducted load testing with Locust.  

## Tools and Technologies

- **Frameworks:** TensorFlow, Keras  
- **Deployment Tools:** ONNX, FastAPI, Locust  
- **Programming Language:** Python  

## How to Use

1. **Dataset:**  
   Download the dataset from [Kaggle](https://www.kaggle.com/datasets/muhammadhananasghar/human-emotions-datasethes) and place it in the appropriate directory.

2. **Running the Code:**  
   Open and run the Jupyter notebook provided (`Human_Emotions_Classification.ipynb`) for training and evaluation.  
   The notebook includes detailed explanations and step-by-step guidance.

3. **Model Deployment:**  
   - Convert the trained model to ONNX format using the code in the notebook.  
   - Use FastAPI scripts to build an API for serving predictions.  

4. **Testing:**  
   Perform load testing using Locust scripts to evaluate the system's performance under various loads.  

## Results

- Achieved a **90% accuracy** on the test set, significantly improving over the baseline accuracy of 71.1%.  
- Interpreted model predictions effectively using Grad-CAM visualizations.  

## Future Improvements

- Extend the model to handle real-time video feeds for emotion recognition.  
- Explore additional datasets to improve model generalization across diverse facial expressions.  
