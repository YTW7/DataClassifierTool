import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from display_visualizations import display_visualizations


def train_and_evaluate_model(
    df,
    feature_columns,
    target_column,
    training_data_percentage
):
    """
    Train and evaluate the selected classification model.
    """

    # ---------------------------------------------------------
    # Separate features (X) and target (Y)
    # ---------------------------------------------------------
    X = df[feature_columns]
    Y = df[target_column]

    # ---------------------------------------------------------
    # Calculate test size
    # ---------------------------------------------------------
    test_size = 1 - (training_data_percentage / 100)

    # ---------------------------------------------------------
    # Split data BEFORE standardization
    # This prevents data leakage.
    # ---------------------------------------------------------
    X_train, X_test, Y_train, Y_test = train_test_split(
        X,
        Y,
        test_size=test_size,
        stratify=Y,
        random_state=2
    )

    # ---------------------------------------------------------
    # Standardize features
    # ---------------------------------------------------------
    st.write("### Standardizing the Features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    # Select classification algorithm
    # ---------------------------------------------------------
    classifier_name = st.selectbox(
        "Choose Classifier",
        [
            "Logistic Regression",
            "Support Vector Machine (SVM)",
            "K-Nearest Neighbors (KNN)",
            "Naive Bayes",
            "Random Forest",
            "Neural Network (MLP)"
        ]
    )

    classifier = select_classifier(classifier_name)

    # ---------------------------------------------------------
    # Train model
    # ---------------------------------------------------------
    st.write("### Training the Classification Model...")

    classifier.fit(X_train_scaled, Y_train)

    # ---------------------------------------------------------
    # Evaluate model
    # ---------------------------------------------------------
    display_model_performance(
        classifier,
        X_train_scaled,
        X_test_scaled,
        Y_train,
        Y_test
    )

    # ---------------------------------------------------------
    # Display random dataset entries
    # ---------------------------------------------------------
    display_random_entries(
        df,
        feature_columns,
        target_column
    )

    # ---------------------------------------------------------
    # Allow user to make new predictions
    # ---------------------------------------------------------
    make_prediction(
        classifier,
        scaler,
        feature_columns
    )

    # ---------------------------------------------------------
    # Generate test predictions
    # ---------------------------------------------------------
    X_test_prediction = classifier.predict(X_test_scaled)

    # ---------------------------------------------------------
    # Display visualizations
    # ---------------------------------------------------------
    display_visualizations(
        df,
        X,
        Y,
        feature_columns,
        target_column,
        X_test_scaled,
        Y_test,
        X_test_prediction
    )


def select_classifier(classifier_name):
    """
    Return the classifier selected by the user.
    """

    if classifier_name == "Logistic Regression":

        return LogisticRegression(
            max_iter=1000
        )

    elif classifier_name == "Support Vector Machine (SVM)":

        return svm.SVC(
            kernel="linear",
            random_state=2
        )

    elif classifier_name == "K-Nearest Neighbors (KNN)":

        return KNeighborsClassifier(
            n_neighbors=5
        )

    elif classifier_name == "Naive Bayes":

        return GaussianNB()

    elif classifier_name == "Random Forest":

        return RandomForestClassifier(
            n_estimators=100,
            random_state=2
        )

    elif classifier_name == "Neural Network (MLP)":

        return MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=500,
            random_state=2
        )


def display_model_performance(
    classifier,
    X_train,
    X_test,
    Y_train,
    Y_test
):
    """
    Display accuracy on training and testing data.
    """

    # ---------------------------------------------------------
    # Training predictions
    # ---------------------------------------------------------
    X_train_prediction = classifier.predict(X_train)

    train_accuracy = accuracy_score(
        Y_train,
        X_train_prediction
    )

    st.write(
        f"Accuracy on Train Data: {train_accuracy:.2f}"
    )

    # ---------------------------------------------------------
    # Testing predictions
    # ---------------------------------------------------------
    X_test_prediction = classifier.predict(X_test)

    test_accuracy = accuracy_score(
        Y_test,
        X_test_prediction
    )

    st.write(
        f"Accuracy on Test Data: {test_accuracy:.2f}"
    )


def display_random_entries(
    df,
    feature_columns,
    target_column
):
    """
    Display one positive and one negative example
    where all selected feature values are non-zero.

    This is mainly useful for the included diabetes dataset.
    """

    diabetic_key = "random_diabetic"
    non_diabetic_key = "random_non_diabetic"

    # ---------------------------------------------------------
    # Filter based on binary target values
    # ---------------------------------------------------------
    non_diabetic_df = df[
        df[target_column] == 0
    ]

    diabetic_df = df[
        df[target_column] == 1
    ]

    # ---------------------------------------------------------
    # Remove rows containing zero feature values
    # ---------------------------------------------------------
    non_diabetic_no_zero = non_diabetic_df[
        (non_diabetic_df[feature_columns] != 0).all(axis=1)
    ]

    diabetic_no_zero = diabetic_df[
        (diabetic_df[feature_columns] != 0).all(axis=1)
    ]

    # ---------------------------------------------------------
    # Check whether valid entries exist
    # ---------------------------------------------------------
    if (
        not non_diabetic_no_zero.empty
        and not diabetic_no_zero.empty
    ):

        # -----------------------------------------------------
        # Store random samples in session state
        # so they don't change on every Streamlit rerun.
        # -----------------------------------------------------
        if diabetic_key not in st.session_state:

            st.session_state[diabetic_key] = (
                diabetic_no_zero.sample(n=1)
            )

        if non_diabetic_key not in st.session_state:

            st.session_state[non_diabetic_key] = (
                non_diabetic_no_zero.sample(n=1)
            )

        # -----------------------------------------------------
        # Display entries
        # -----------------------------------------------------
        st.write(
            "### Entries from Dataset for testing:"
        )

        st.write("#### Positive:")

        st.write(
            st.session_state[diabetic_key][
                feature_columns + [target_column]
            ]
        )

        st.write("#### Negative:")

        st.write(
            st.session_state[non_diabetic_key][
                feature_columns + [target_column]
            ]
        )

    else:

        st.write(
            "No valid entries with non-zero feature "
            "values found for both diabetic and "
            "non-diabetic groups."
        )


def make_prediction(
    classifier,
    scaler,
    feature_columns
):
    """
    Take user input and generate a prediction
    using the trained classifier.
    """

    st.write("### Make a New Prediction")

    user_input = []

    # ---------------------------------------------------------
    # Create an input field for every selected feature
    # ---------------------------------------------------------
    for feature in feature_columns:

        value = st.number_input(
            f"Enter value for {feature}",
            step=0.1
        )

        user_input.append(value)

    # ---------------------------------------------------------
    # Prediction button
    # ---------------------------------------------------------
    if st.button("Predict"):

        input_data_as_numpy_array = np.asarray(
            user_input
        ).reshape(1, -1)

        # Apply the SAME scaler fitted on training data
        standardized_data = scaler.transform(
            input_data_as_numpy_array
        )

        prediction = classifier.predict(
            standardized_data
        )

        # -----------------------------------------------------
        # Display result
        # -----------------------------------------------------
        if prediction[0] == 0:

            st.success(
                "The model predicts a **negative outcome** "
                "(no condition detected)."
            )

        else:

            st.warning(
                "The model predicts a **positive outcome** "
                "(condition detected)."
            )
