import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier

# -----------------------------
# Load Trained Model
# -----------------------------
model = XGBClassifier()
model.load_model("diabetes_model.json")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    layout="centered"
)

# -----------------------------
# Title
# -----------------------------
st.title("Diabetes Prediction System")

st.subheader(
    "An Explainable Intelligent Diabetes Risk Prediction and Recommendation System"
)

st.write("Enter the health parameters below:")

# -----------------------------
# User Inputs
# -----------------------------

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=1,
    max_value=100,
    value=25
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1]
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1]
)

smoking_history = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "not current"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=22.0
)

HbA1c_level = st.number_input(
    "HbA1c Level",
    min_value=3.0,
    max_value=15.0,
    value=5.0
)

blood_glucose_level = st.number_input(
    "Blood Glucose Level",
    min_value=50,
    max_value=300,
    value=100
)

# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    # Create input dataframe
    input_dict = {

        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level,

        # Gender Encoding
        'gender_Male': 1 if gender == "Male" else 0,

        # Smoking Encoding
        'smoking_history_former': 1 if smoking_history == "former" else 0,
        'smoking_history_never': 1 if smoking_history == "never" else 0,
        'smoking_history_not current': 1 if smoking_history == "not current" else 0
    }

    input_data = pd.DataFrame([input_dict])

    # Prediction
    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # -----------------------------
    # Prediction Result
    # -----------------------------

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Probability of Diabetes")
    else:
        st.success("Low Probability of Diabetes")

    # -----------------------------
    # Risk Level
    # -----------------------------

    if probability < 0.3:
        risk = "Low Risk"

    elif probability < 0.7:
        risk = "Medium Risk"

    else:
        risk = "High Risk"

    st.subheader(f"Risk Level: {risk}")

    st.write(
        f"Prediction Probability: {probability * 100:.2f}%"
    )

    # -----------------------------
    # Recommendations
    # -----------------------------

    st.subheader("Personalized Recommendations")

    recommendations = []

    if HbA1c_level > 6.5:
        recommendations.append(
            "Monitor blood sugar levels regularly."
        )

    if blood_glucose_level > 140:
        recommendations.append(
            "Reduce sugar intake and maintain balanced diet."
        )

    if bmi > 30:
        recommendations.append(
            "Maintain regular exercise and healthy eating habits."
        )

    if hypertension == 1:
        recommendations.append(
            "Monitor blood pressure regularly."
        )

    if heart_disease == 1:
        recommendations.append(
            "Consult doctor for regular cardiovascular checkups."
        )

    if smoking_history != "never":
        recommendations.append(
            "Avoid smoking and maintain healthy lifestyle habits."
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Maintain healthy lifestyle and regular checkups."
        )

    for rec in recommendations:
        st.write("- ", rec)
