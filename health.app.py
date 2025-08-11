import streamlit as st
import pandas as pd
import time
import pickle

# Load the model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Load the scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Personalized Healthcare Recommendations", page_icon="🩺", layout="centered")
st.title('🩺 Personalized Healthcare Recommendations Prediction')
st.markdown("**Machine Learning Model: LogisticRegression**")
st.info("Adjust the input parameters in the sidebar to see the prediction in real time.")

# Default values (replace with raw unscaled values from inverse transform)
low_risk_values = {"Recency": 4.0, "Frequency": 7.0, "Monetary": 1750.0, "Time": 25.0}
high_risk_values = {"Recency": 2.0, "Frequency": 10.0, "Monetary": 2500.0, "Time": 49.0}

# Sidebar inputs
with st.sidebar:
    st.header('📊 Input Features')

    # Quick-set buttons
    if st.button("Set Low Risk Example"):
        Recency = low_risk_values["Recency"]
        Frequency = low_risk_values["Frequency"]
        Monetary = low_risk_values["Monetary"]
        Time = low_risk_values["Time"]
    elif st.button("Set High Risk Example"):
        Recency = high_risk_values["Recency"]
        Frequency = high_risk_values["Frequency"]
        Monetary = high_risk_values["Monetary"]
        Time = high_risk_values["Time"]
    else:
        Recency = st.slider('Recency', 0.0, 26.0)
        Frequency = st.slider('Frequency', 1.0, 12.0)
        Monetary = st.slider('Monetary', 250.0, 3000.0)
        Time = st.slider('Time', 0.0, 50.0)  # Adjusted to your range

# Predict button
if st.button("🚀 Predict"):
    data = pd.DataFrame([[Recency, Frequency, Monetary, Time]],
                        columns=['Recency', 'Frequency', 'Monetary', 'Time'])

    # Apply scaling
    data_scaled = scaler.transform(data)
    predictions = best_model.predict(data_scaled)

    with st.spinner('Calculating prediction...'):
        time.sleep(1)

    st.success("✅ Prediction Completed!")

    # Animated display cards
    if predictions[0] == 0:
        st.markdown("""
        <div style="background-color:#e8f5e9;padding:20px;border-radius:15px;text-align:center;
                    animation: fadeIn 1s;">
            <h2 style="color:#2e7d32;">🟢 Low-Risk Health Category</h2>
            <p style="font-size:18px;">You are in a low-risk category. Keep up your healthy lifestyle!</p>
        </div>
        <style>
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
        </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color:#ffebee;padding:20px;border-radius:15px;text-align:center;
                    animation: pulse 1s infinite;">
            <h2 style="color:#c62828;">🔴 High-Risk Health Category</h2>
            <p style="font-size:18px;">You may be at higher risk. We recommend a professional health check-up.</p>
        </div>
        <style>
            @keyframes pulse {
                0% { box-shadow: 0 0 0 0 rgba(198,40,40, 0.7); }
                70% { box-shadow: 0 0 0 10px rgba(198,40,40, 0); }
                100% { box-shadow: 0 0 0 0 rgba(198,40,40, 0); }
            }
        </style>
        """, unsafe_allow_html=True)

    st.metric(label="📊 Predicted Health Status", value="Low Risk" if predictions[0] == 0 else "High Risk")

