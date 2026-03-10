import streamlit as st
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("models/lstm_model.keras", compile=False)

st.title("AI Credit Card Fraud Detection")

st.write("Enter 30 transaction features separated by commas")

user_input = st.text_input("Transaction Data")

if st.button("Predict"):

    try:
        values = [float(x) for x in user_input.split(",")]

        if len(values) != 30:
            st.error("Please enter exactly 30 values")
        else:
            data = np.array(values).reshape(1,10,3)

            prediction = model.predict(data)

            risk_score = float(prediction[0][0])

            if risk_score > 0.5:
                st.error(f"Fraud Detected 🚨 | Risk Score: {risk_score:.4f}")
            else:
                st.success(f"Transaction Safe ✅ | Risk Score: {risk_score:.4f}")

    except:
        st.error("Invalid input format")