import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from display_visualizations import display_visualizations 

def train_and_evaluate_model(df, feature_columns, target_column, training_data_percentage):

    """Train the selected model and evaluate its performance."""
    X = df[feature_columns]
    Y = df[target_column]
    
    # Standardize features
    st.write("### Standardizing the Features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    users_test_size = 1 - (training_data_percentage / 100)
    
    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=users_test_size, stratify=Y, random_state=2)
    
    # Select classification algorithm
    classifier_name = st.selectbox("Choose Classifier", ["Logistic Regression", "SVM (Support Vector Machine)"])
    classifier = select_classifier(classifier_name)
    
    # Train the model
    st.write("### Training the Classification Model...")
    classifier.fit(X_train, Y_train)
    
    # Evaluate the model
    display_model_performance(classifier, X_train, X_test, Y_train, Y_test, df, feature_columns, target_column)
    
    # Random entries
    display_random_entries(df, feature_columns, target_column)

    # Allow user to make predictions
    make_prediction(classifier, scaler, feature_columns)

    X_test_prediction = classifier.predict(X_test)
    display_visualizations(df, X_scaled, Y, feature_columns, target_column, X_test, Y_test, X_test_prediction)


def select_classifier(classifier_name):
    """Select and return the appropriate classifier based on user input."""
    if classifier_name == "Logistic Regression":
        return LogisticRegression()
    else:
        return svm.SVC(kernel='linear', random_state=2)
    
def display_model_performance(classifier, X_train, X_test, Y_train, Y_test, df, feature_columns, target_column):
    """Display accuracy on training and test data."""
    X_train_prediction = classifier.predict(X_train)
    train_accuracy = accuracy_score(Y_train, X_train_prediction)
    st.write(f"Accuracy on Train Data: {train_accuracy:.2f}")
    
    X_test_prediction = classifier.predict(X_test)
    test_accuracy = accuracy_score(Y_test, X_test_prediction)
    st.write(f"Accuracy on Test Data: {test_accuracy:.2f}")

# Function to fetch and display two random entries where no feature value is zero
def display_random_entries(df, feature_columns, target_column):
    # Define session state keys
    diabetic_key = "random_diabetic"
    non_diabetic_key = "random_non_diabetic"

    # Filter dataset for both outcome = 0 (non-diabetic) and outcome = 1 (diabetic)
    non_diabetic_df = df[df[target_column] == 0]
    diabetic_df = df[df[target_column] == 1]
    
    # Filter out rows where any feature is zero
    non_diabetic_no_zero = non_diabetic_df[(non_diabetic_df[feature_columns] != 0).all(axis=1)]
    diabetic_no_zero = diabetic_df[(diabetic_df[feature_columns] != 0).all(axis=1)]
    
    # Check if there are any valid entries
    if not non_diabetic_no_zero.empty and not diabetic_no_zero.empty:
        # Check if random entries are already stored in session state
        if diabetic_key not in st.session_state:
            st.session_state[diabetic_key] = diabetic_no_zero.sample(n=1)
        if non_diabetic_key not in st.session_state:
            st.session_state[non_diabetic_key] = non_diabetic_no_zero.sample(n=1)
        
        # Display the stored entries
        st.write("### Entries from Dataset for testing:")
        st.write("#### Positive:")
        st.write(st.session_state[diabetic_key][feature_columns + [target_column]])
        st.write("#### Negative:")
        st.write(st.session_state[non_diabetic_key][feature_columns + [target_column]])
    else:
        st.write("No valid entries with non-zero feature values found for both diabetic and non-diabetic groups.")


def make_prediction(classifier, scaler, feature_columns):
    """Take user input for testing and make predictions."""
    st.write("### Make a New Prediction")
    user_input = []
    for feature in feature_columns:
        
        # Allow user to enter a value within this range
        value = st.number_input(f"Enter value for {feature}", step=0.1)
        user_input.append(value)
    
    if st.button("Predict"):
        input_data_as_numpy_array = np.asarray(user_input).reshape(1, -1)
        std_data = scaler.transform(input_data_as_numpy_array)
        prediction = classifier.predict(std_data)
        
        if prediction[0] == 0:
          st.success("The model predicts a **negative outcome** (no condition detected).")
        else:
          st.warning("The model predicts a **positive outcome** (condition detected).")
