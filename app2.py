import streamlit as st
import pandas as pd
import joblib

# ---------- Page setup ----------
st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

# ---------- Fixed values for features removed from the form ----------
# The model was trained with these 2 features, so it still needs a value for them.
# They are no longer asked in the form; these defaults (most common values in
# heart.csv) are used instead. Change them here if you want different defaults.
DEFAULT_EXERCISE_ANGINA = "No"    # "Yes" or "No"
DEFAULT_ST_SLOPE = "Flat"         # "Up", "Flat" or "Down"


# ---------- Load saved files (only once, then cached) ----------
@st.cache_resource
def load_files():
    model = joblib.load("KNN_heart.pkl")
    scaler = joblib.load("scaler.pkl")
    columns = joblib.load("columns.pkl")
    return model, scaler, columns


try:
    model, scaler, columns = load_files()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Keep app.py, KNN_heart.pkl, scaler.pkl and columns.pkl in the same folder.")
    st.stop()

# ---------- Title ----------
st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's health information to predict the likelihood of heart disease.")
st.divider()

# ---------- Input form ----------
st.subheader("Patient Information")

# Age as a slider
age = st.slider("Age", min_value=1, max_value=120, value=50)

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Sex", ["Male", "Female"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=250,
                                 value=None, step=1, placeholder="e.g. 120")
    fasting_sugar = st.number_input("Fasting Blood Sugar (mg/dl)", min_value=0, max_value=600,
                                    value=None, step=1, placeholder="e.g. 100")
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])

with col2:
    cholesterol = st.number_input("Cholesterol (mg/dl)", min_value=0, max_value=700,
                                  value=None, step=1, placeholder="e.g. 200")
    max_hr = st.number_input("Maximum Heart Rate", min_value=40, max_value=250, value=150, step=1)
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    oldpeak = st.number_input("Oldpeak", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)

# ---------- Predict button ----------
if st.button("Predict", type="primary", use_container_width=True):

    # Basic validation
    errors = []
    if resting_bp is None:
        errors.append("Please enter Resting Blood Pressure.")
    elif resting_bp == 0:
        errors.append("Resting Blood Pressure cannot be 0.")
    if cholesterol is None:
        errors.append("Please enter Cholesterol.")
    if fasting_sugar is None:
        errors.append("Please enter Fasting Blood Sugar.")
    if max_hr < 60:
        errors.append("Maximum Heart Rate should be at least 60.")

    if errors:
        for msg in errors:
            st.error(msg)
    else:
        # Step 1: convert inputs to the 15 model features (one-hot encoding)
        input_data = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": 1 if fasting_sugar > 120 else 0,          # > 120 mg/dl -> 1
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_M": 1 if sex == "Male" else 0,
            "ChestPainType_ATA": 1 if chest_pain == "ATA" else 0,
            "ChestPainType_NAP": 1 if chest_pain == "NAP" else 0,
            "ChestPainType_TA": 1 if chest_pain == "TA" else 0,    # ASY -> all three 0
            "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
            "RestingECG_ST": 1 if resting_ecg == "ST" else 0,       # LVH -> both 0
            "ExerciseAngina_Y": 1 if DEFAULT_EXERCISE_ANGINA == "Yes" else 0,
            "ST_Slope_Flat": 1 if DEFAULT_ST_SLOPE == "Flat" else 0,
            "ST_Slope_Up": 1 if DEFAULT_ST_SLOPE == "Up" else 0,    # Down -> both 0
        }

        # Step 2: build DataFrame in the exact order saved in columns.pkl
        input_df = pd.DataFrame([input_data])[columns]

        # Step 3: scale, then predict
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        # ---------- Result ----------
        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.error("Heart Disease Detected")
        else:
            st.success("No Heart Disease Detected")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(scaled_input)[0][1] * 100
            st.write(f"**Predicted Probability of Heart Disease: {probability:.1f}%**")
            st.progress(int(round(probability)))

# ---------- Disclaimer ----------
st.divider()
st.caption("This application is for educational and demonstration purposes only and is not a medical diagnosis.")
