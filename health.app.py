import streamlit as st
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import pickle

# Load the model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

# Load the Scaler 
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.set_page_config(page_title="Personlized Healthcare Recommendations", page_icon="🩺", layout="centered")
st.title('🩺 ersonlized Healthcare Recommendations Prediction')
st.markdown("**Machine Learning Model: LogisticRegression**")
st.info("Adjust the input parameters in the sidebar to see the prediction in real time.")


# Sidebar inputs
with st.sidebar:
    st.header('📊 Input Features')
    Recency = st.slider('Recency', 0., 26.)
    Frequency = st.slider('Frequency', 1., 12.)
    Monetary = st.slider('Monetary', 250., 3000.)
    Time = st.slider('Time', 2,, 99.)

# Predict button
if st.button("🚀 Predict"):
    data = pd.DataFrame([[Recency, Frequency, Monetary, Time]],
                        columns=['Recency', 'Frequency', 'Monetary', 'Time'])

    data_scaled = scaler.transform(data)
    predictions = best_model.predict(data_scaled)

    # Simulate a loading animation
    with st.spinner('Calculating prediction...'):
        time.sleep(1)


    # Mapping predictions to messages
    if predictions[0] == 0:
        message = "🟢 **Low-Risk Health Category** – You are currently in a low-risk group. Keep up your healthy habits!"
        bg_color = "#e8f5e9"  # Light green
        text_color = "#2e7d32"
    else:
        message = "🔴 **High-Risk Health Category** – You may be at higher risk. Please consider a professional health consultation."
        bg_color = "#ffebee"  # Light red
        text_color = "#c62828"

    # Stylish display
    st.success("✅ Prediction Completed!")
    st.markdown(
        f"""
        <div style="background-color:{bg_color};padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:{text_color};">Prediction Result</h2>
            <p style="font-size:20px;color:{text_color};">{message}</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Optional: Metric display
    st.metric(label="📊 Predicted Health Status", value="Low Risk" if predictions[0] == 0 else "High Risk")
