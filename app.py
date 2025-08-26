import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------
# Load trained model (or train a quick one if no saved model)
# ---------------------------

@st.cache_resource
def load_model():
    try:
        with open("svm_model.pkl", "rb") as f:
            model, scaler = pickle.load(f)
    except:
        # If model not found, create a dummy model (for demo)
        from sklearn.model_selection import train_test_split
        from sklearn.datasets import make_classification

        X, y = make_classification(n_samples=500, n_features=9, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = SVC(probability=True, kernel="rbf")
        model.fit(X_scaled, y)

        with open("svm_model.pkl", "wb") as f:
            pickle.dump((model, scaler), f)

    return model, scaler

model, scaler = load_model()

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("💧 Water Potability Prediction")
st.write("Enter the water quality parameters to predict if water is safe to drink.")

# Input fields
pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
hardness = st.number_input("Hardness", min_value=0.0, value=200.0, step=1.0)
solids = st.number_input("Solids (ppm)", min_value=0.0, value=10000.0, step=100.0)
chloramines = st.number_input("Chloramines (ppm)", min_value=0.0, value=7.0, step=0.1)
sulfate = st.number_input("Sulfate (mg/L)", min_value=0.0, value=300.0, step=1.0)
conductivity = st.number_input("Conductivity (μS/cm)", min_value=0.0, value=400.0, step=1.0)
organic_carbon = st.number_input("Organic Carbon (ppm)", min_value=0.0, value=15.0, step=0.1)
trihalomethanes = st.number_input("Trihalomethanes (μg/L)", min_value=0.0, value=60.0, step=1.0)
turbidity = st.number_input("Turbidity (NTU)", min_value=0.0, value=4.0, step=0.1)

# Prediction
if st.button("Predict Potability"):
    input_data = np.array([[pH, hardness, solids, chloramines, sulfate,
                            conductivity, organic_carbon, trihalomethanes, turbidity]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][prediction] * 100

    if prediction == 1:
        st.success(f"✅ Water is Potable (Safe to Drink) - Confidence: {prob:.2f}%")
    else:
        st.error(f"❌ Water is Not Potable (Unsafe) - Confidence: {prob:.2f}%")
