# Explainable Ensemble Learning for Cardiovascular Risk Prediction

## 📌 Project Overview

This project presents an **Explainable Ensemble Learning Framework for Cardiovascular Risk Prediction**. It uses machine learning techniques to predict cardiovascular disease risk based on patient clinical and health-related parameters.

The system combines **Random Forest** and **Gradient Boosting** using a soft-voting ensemble approach to improve prediction performance. Explainable AI techniques are used to identify the important factors influencing the prediction.

## 🎯 Objectives

- Predict cardiovascular disease risk using machine learning.
- Combine multiple ML models using ensemble learning.
- Identify important factors influencing cardiovascular risk.
- Provide understandable explanations for model predictions.
- Classify patients according to their risk level.
- Provide personalized preventive healthcare recommendations.
- Develop an interactive Streamlit web application.

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Explainable AI (XAI)
- Random Forest
- Gradient Boosting
- Soft Voting Ensemble

## 📊 Dataset

The project uses the **Heart Disease Dataset** available on Kaggle.

**Dataset Source:**  
https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

The dataset contains **1,025 patient records** with 13 clinical features and a target variable.

### Features

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- Oldpeak
- Slope
- Number of Major Vessels
- Thalassemia

## ⚙️ Methodology

The project follows these major steps:

1. Data Collection
2. Data Preprocessing
3. Feature Preparation
4. Model Training
5. Ensemble Model Development
6. Model Evaluation
7. Explainable AI Analysis
8. Risk Assessment
9. Personalized Recommendations
10. Streamlit Deployment

## 🤖 Machine Learning Models

### Random Forest

Random Forest is used to build multiple decision trees and combine their predictions to achieve reliable classification performance.

### Gradient Boosting

Gradient Boosting builds models sequentially, where each new model focuses on correcting errors made by previous models.

### Soft Voting Ensemble

The predictions from Random Forest and Gradient Boosting are combined using a **Soft Voting Ensemble** to obtain the final prediction.

## 🔍 Explainable AI

The system provides explanations for cardiovascular risk predictions by analyzing feature importance.

Important clinical factors may include:

- Chest Pain Type
- Maximum Heart Rate
- Number of Major Vessels
- Exercise-Induced Angina
- Cholesterol
- Resting Blood Pressure
- Age

This helps users understand which factors contribute to the model's prediction.

## 📈 Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Feature Importance

The ensemble model achieved approximately **97.56% accuracy** on the evaluated dataset.

## 🖥️ Application Features

The Streamlit application provides:

- 👤 Patient Details
- 🆔 Unique Patient ID
- 📊 Cardiovascular Risk Assessment
- 📈 Risk Gauge
- 🔍 Explainable AI Feature Importance
- ❤️ Risk Factors
- 💡 Personalized Recommendations
- 📋 Assessment History
- 📄 Downloadable PDF Medical Report

## 📂 Project Structure

```text
Cardiovascular_Disease_Prediction/
│
├── app.py
├── heart_model.pkl
├── feature_importance.csv
├── README.md
└── requirements.txt
