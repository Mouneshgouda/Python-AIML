# app.py
import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler

clf = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


st.title("Placement Prediction App")

cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
iq = st.number_input("IQ", min_value=50, max_value=200, value=100, step=1)


if st.button("Predict"):
    X_new = np.array([[cgpa, iq]])
    X_scaled = scaler.transform(X_new)
    pred = clf.predict(X_scaled)
    st.success("Placed 🎉" if pred[0]==1 else "Not Placed 😔")
