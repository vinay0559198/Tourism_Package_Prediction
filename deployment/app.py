import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load model
MODEL_REPO = "vinay9700/tourism-package-prediction-model"
MODEL_FILE = "model.pkl"

model_path = hf_hub_download(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    repo_type="model"
)

model = joblib.load(model_path)

st.title("Tourism Package Prediction")

# Collect inputs
age = st.number_input("Age", 18, 100, 35)
city = st.selectbox("City Tier", [1, 2, 3])
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Monthly Income", 1000, 200000, 45000)

if st.button("Predict"):
    user_input = {
        "Age": age,
        "TypeofContact": "Self Enquiry",
        "CityTier": city,
        "DurationOfPitch": 15,
        "Gender": gender,
        "NumberOfPersonVisiting": 2,
        "NumberOfChildrenVisiting": 0,
        "NumberOfFollowups": 3,
        "ProductPitched": "Basic",
        "PreferredPropertyStar": 3,
        "MaritalStatus": "Married",
        "NumberOfTrips": 2,
        "Passport": 1,
        "PitchSatisfactionScore": 4,
        "OwnCar": 1,
        "Occupation": "Salaried",
        "Designation": "Executive",
        "MonthlyIncome": income,
        "Unnamed: 0": 0
    }

    df = pd.DataFrame([user_input])
    prediction = model.predict(df)[0]

    if prediction == 1:
        st.success("✅ Customer is likely to take the package")
    else:
        st.warning("❌ Customer is unlikely to take the package")
