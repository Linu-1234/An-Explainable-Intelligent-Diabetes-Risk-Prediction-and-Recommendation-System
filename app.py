import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Page title
st.set_page_config(page_title="An-Explainable-Intelligent-Diabetes-Risk-Prediction-and-Recommendation-System")

st.title("An-Explainable-Intelligent-Diabetes-Risk-Prediction-and-Recommendation-System")

st.write("Enter the health parameters below:")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])

age = st.number_input("Age", 1, 100)

hypertension = st.selectbox("Hypertension", [0,1])

heart_disease = st.selectbox("Heart Disease", [0,1])

smoking_history = st.selectbox(
    "Smoking History",
    ["never", "former", "current", "not current"]
)

bmi = st.number_input("BMI", 10.0, 60.0)

HbA1c_level = st.number_input("HbA1c Level", 3.0, 15.0)

blood_glucose_level = st.number_input(
    "Blood Glucose Level",
    50,
    300
)

# Encoding
gender_Male = 1 if gender == "Male" else 0

smoking_history_former = 1 if smoking_history == "former" else 0
smoking_history_never = 1 if smoking_history == "never" else 0
smoking_history_not_current = 1 if smoking_history == "not current" else 0

# Prediction button
if st.button("Predict"):

    input_data = np.array([[
        age,
        hypertension,
        heart_disease,
        bmi,
        HbA1c_level,
        blood_glucose_level,
        gender_Male,
        smoking_history_former,
        smoking_history_never,
        smoking_history_not_current
    ]])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    # Prediction Result
    if prediction == 1:
        st.error("High Probability of Diabetes")
    else:
        st.success("Low Probability of Diabetes")

    # Risk Level
    if probability < 0.3:
        risk = "Low Risk"

    elif probability < 0.7:
        risk = "Medium Risk"

    else:
        risk = "High Risk"

    st.subheader(f"Risk Level: {risk}")

    st.write(f"Prediction Probability: {probability:.2f}")

    # Recommendations
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