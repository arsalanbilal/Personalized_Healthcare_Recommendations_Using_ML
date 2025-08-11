import streamlit as st
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
import pickle

# Load the model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)

st.set_page_config(page_title="Personlized Healthcare Recommendations", page_icon="🩺", layout="centered")
st.title('🩺 ersonlized Healthcare Recommendations Prediction')
st.markdown("**Machine Learning Model: LogisticRegression**")
st.info("Adjust the input parameters in the sidebar to see the prediction in real time.")


# Sidebar inputs
with st.sidebar:
    st.header('📊 Input Features')
    Recency = st.slider('Recency', 0.0, 26.0)
    Frequency = st.slider('Frequency', 1.0, 12.0)
    Monetary = st.slider('Monetary', 250.0, 3000.0)
    Time = st.slider('Time', 0.0, 1.0)

# Predict button
    if st.button("🚀 Predict"):
    data = pd.DataFrame([[Recency, Frequency, Monetary, Time]],
                        columns=['Recency', 'Frequency', 'Monetary', 'Time'])

    predictions = best_model.predict(data)

    # Simulate a loading animation
    with st.spinner('Calculating prediction...'):
        time.sleep(1)

    # Stylish display
    st.success("✅ Prediction Completed!")
    st.markdown(
        f"""
        <div style="background-color:#f0f8ff;padding:20px;border-radius:10px;text-align:center;">
            <h2 style="color:#1E90FF;">Predicted Value</h2>
            <h1 style="color:#FF4500;font-size:60px;">{predictions[0]:,.2f}</h1>
            <p style="color:gray;">Estimated Uber demand based on provided features</p>
        </div>
        """, unsafe_allow_html=True
    )

if predictions[0] ==1:
    st.success("You are likely to have Diabities.")
else:
    st.success("You are Unlikely to have Diabities")  

    # Optional: Metric display
    st.metric(label="📈 Predicted Health Status", value=f"{predictions[0]:,.2f}")
