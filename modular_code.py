import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt
from train_and_evaluate_model import train_and_evaluate_model 


def main():
    setup_page()
    st.title("Disease Detector App")
    st.write("Upload your dataset, train a Classification model, and make Predictions.")
    
    st.download_button(
        label="Download Sample Dataset (diabetes.csv)",
        data=open('./diabetes.csv', 'rb').read(),  # Opens the diabetes.csv file and reads it
        file_name="diabetes.csv",  # Name of the file that will be downloaded
        mime="text/csv"  # MIME type for CSV file
    )

    uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])
    if uploaded_file is not None:
        df = load_dataset(uploaded_file)
        num_rows, num_cols = df.shape
        st.write(f"Dataset contains {num_rows} rows and {num_cols} columns.")
        feature_columns, target_column = select_features_and_target(df)
        if feature_columns and target_column:
            training_data_percentage = st.slider(
                "Select percentage of training data", 
                min_value=0, 
                max_value=100, 
                value=80,  # default is 80% for training data
                step=5
            )
            train_and_evaluate_model(df, feature_columns, target_column, training_data_percentage)


def setup_page():
    """Set up the Streamlit page configuration."""
    st.set_page_config(
        page_title="Disease Detector App",
        page_icon="./docicon.png",
        # layout="wide"
    )


def load_dataset(file):
    """Load the uploaded dataset and display a preview."""
    df = pd.read_csv(file)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
    return df


def select_features_and_target(df):
    """Allow the user to select features (X) and target (Y) from the dataset."""
    st.write("### Select Features (X) and Target (Y)")
    all_columns = df.columns.tolist()
    target_column = st.selectbox("Select Target Column (Y)", all_columns)
    feature_columns = st.multiselect("Select Feature Columns (X)", [col for col in all_columns if col != target_column])
    return feature_columns, target_column

if __name__ == "__main__":
    main()