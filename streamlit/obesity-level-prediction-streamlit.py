import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Obesity Prediction", page_icon="🏥")

st.title("🏥 Obesity Level Prediction")
st.write("Aplikasi prediksi obesity level - Testing Mode")

# Input sederhana dulu
age = st.slider("Age", 14, 61, 25)
height = st.slider("Height (m)", 1.45, 1.98, 1.70)
weight = st.slider("Weight (kg)", 39, 173, 70)

if st.button("Calculate BMI"):
    bmi = weight / (height ** 2)
    st.write(f"**BMI:** {bmi:.2f}")
    
    if bmi < 18.5:
        category = "Insufficient Weight"
    elif bmi < 25:
        category = "Normal Weight" 
    elif bmi < 30:
        category = "Overweight Level I"
    elif bmi < 35:
        category = "Obesity Type I"
    else:
        category = "Obesity Type II/III"
        
    st.success(f"**Predicted Category:** {category}")
