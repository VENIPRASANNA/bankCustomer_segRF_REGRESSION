import streamlit as st
import numpy as np
import pickle
import os

# Page config
st.set_page_config(
    page_title="Bank Customer Segmentation",
    page_icon="🏦",
    layout="centered"
)

# Load model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

kmeans = pickle.load(open(os.path.join(BASE_DIR, "kmeans_model.pkl"), "rb"))
scaler = pickle.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))

# App title
st.title("🏦 Bank Customer Segmentation")
st.caption("K-Means Clustering (Unsupervised Learning)")
st.markdown("---")

# Inputs
credit_score = st.slider("Credit Score", 300, 900, 650)
age = st.slider("Age", 18, 80, 40)
tenure = st.slider("Tenure (years)", 0, 10, 5)
balance = st.number_input("Account Balance", value=50000.0)
products = st.slider("Number of Products", 1, 4, 2)
salary = st.number_input("Estimated Salary", value=60000.0)

# Predict cluster
if st.button("Find Customer Cluster"):
    data = np.array([[credit_score, age, tenure, balance, products, salary]])
    data_scaled = scaler.transform(data)
    cluster = kmeans.predict(data_scaled)[0]

    st.success(f"🧠 Customer belongs to **Cluster {cluster}**")
