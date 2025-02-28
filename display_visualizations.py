import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import train_and_evaluate_model as train_and_evaluate_model
import seaborn as sns
import matplotlib.pyplot as plt

def display_visualizations(df, X, Y, feature_columns, target_column, X_test, Y_test, predictions):
    """Allow the user to choose one visualization at a time and display it."""
    st.write("### Visualization Options")
    
    # Let the user select one visualization type
    visualization_option = st.radio(
        "Choose a visualization to display:",
        ["Feature Distribution","Heatmap (Confusion Matrix)"]
    )
    
    # Display the selected visualization
    if visualization_option == "Feature Distribution":
        visualize_feature_distribution(df, feature_columns, target_column)
    elif visualization_option == "Heatmap (Confusion Matrix)":
        display_confusion_matrix(Y_test, predictions)


def display_confusion_matrix(Y_test, predictions):
    """Display a heatmap of the confusion matrix with customized colors."""
    
    # Calculate the confusion matrix
    cm = confusion_matrix(Y_test, predictions)
    
    # Extract TP, TN, FP, FN from the confusion matrix
    tn, fp, fn, tp = cm.ravel()
 
    # Create the heatmap plot
    st.write("### Confusion Matrix Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'], ax=ax, annot_kws={'size': 14})

    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")

    st.pyplot(fig)

    # Display the extracted metrics
    st.write(f"**True Positive (TP)**: {tp} Correctly predicted positive cases.")
    st.write(f"**True Negative (TN)**: {tn} Correctly predicted negative cases.")
    st.write(f"**False Positive (FP)**: {fp} Cases incorrectly predicted as positive.")
    st.write(f"**False Negative (FN)**: {fn} Cases incorrectly predicted as negative.")
    st.write(f"**The classification model achieved an accuracy of: {((tp + tn) / (tp + tn + fp + fn)) * 100:.2f}%**")

def visualize_feature_distribution(df, feature_columns, target_column):

    st.write("### Feature Distribution by Class")
    for feature in feature_columns:
        fig, ax = plt.subplots()
        sns.kdeplot(data=df, x=feature, hue=target_column, fill=True, common_norm=False, ax=ax)
        ax.set_title(f"Distribution of {feature} by {target_column}")
        st.pyplot(fig)

