import streamlit as st
import pandas as pd
import time
import pickle

# Load the model
with open('best_model.pkl', 'rb') as f:
    best_model = pickle.load(f)



st.set_page_config(page_title="Personalized Healthcare Recommendations", page_icon="🩺", layout="centered")
st.title('🩺 Personalized Healthcare Recommendations Prediction')
st.markdown("**Machine Learning Model: LogisticRegression**")
st.info("Adjust the input parameters in the sidebar to see the prediction in real time.")



# Sidebar inputs
with st.sidebar:
    st.header('📊 Input Features')
    Recency = st.slider('Recency', 
 

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

