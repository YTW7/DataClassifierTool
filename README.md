# Interactive Machine Learning Classification & Prediction Platform

A Streamlit-based machine learning application that enables users to upload tabular datasets, select feature and target variables, configure the training/testing split, train classification models, evaluate their performance, visualize results, and generate predictions on new data.

The application provides an interactive interface for experimenting with supervised classification workflows without requiring users to write machine learning code for every dataset.

## Features

- **Dataset Upload**
  - Upload custom CSV datasets through the Streamlit interface.
  - Preview the uploaded dataset and inspect its dimensions.

- **Feature & Target Selection**
  - Select any column as the target variable.
  - Select one or more columns as input features.

- **Configurable Train/Test Split**
  - Choose the percentage of data used for training through an interactive slider.

- **Classification Algorithms**
  - Logistic Regression
  - Support Vector Machine (SVM)

- **Feature Standardization**
  - Standardizes numerical features using `StandardScaler`.

- **Model Evaluation**
  - Training accuracy
  - Testing accuracy
  - Confusion matrix
  - True Positives (TP)
  - True Negatives (TN)
  - False Positives (FP)
  - False Negatives (FN)

- **Interactive Visualizations**
  - Feature distributions by target class
  - Confusion matrix heatmap

- **New Data Prediction**
  - Enter feature values manually.
  - Generate predictions using the trained classification model.

- **Sample Dataset**
  - Includes a sample `diabetes.csv` dataset for testing the application.

## Application Workflow

```text
                ┌──────────────────────┐
                │     Upload Dataset   │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Preview Dataset      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Select Features (X)  │
                │ Select Target (Y)    │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Configure Train/Test │
                │ Split                │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Select Classifier    │
                │ • Logistic Regression│
                │ • SVM                │
                │ • KNN                │
                │ • NeuralNetwork      │
                │ • Random Forest      │
                └──────────┬───────────┘
                           │
                           ▼
                ┌──────────────────────┐
                │ Standardize Features │
                │ & Train Model        │
                └──────────┬───────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │ Model Evaluation         │
              │ • Train Accuracy         │
              │ • Test Accuracy          │
              │ • Confusion Matrix       │
              └────────────┬─────────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
       ┌──────────────────┐  ┌──────────────────┐
       │ Visualize Data   │  │ New Prediction   │
       │ & Results        │  │ Using Model      │
       └──────────────────┘  └──────────────────┘
