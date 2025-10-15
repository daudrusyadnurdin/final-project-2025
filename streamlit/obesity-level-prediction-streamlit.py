import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# --------------------------
# Load model XGBoost dari URL
# --------------------------
MODEL_URL = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/streamlit/xgb-obesity.json"
response = requests.get(MODEL_URL)
model_json = response.text

model = xgb.XGBClassifier()
model.load_model(json.loads(model_json))

# --------------------------
# Encode kategori ke angka (sesuai data training)
# --------------------------
gender_map = {"Male": 1, "Female": 0}
yn_map = {"yes": 1, "no": 0}
favc_map = {"yes": 1, "no": 0}
caec_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
calc_map = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
mtrans_map = {"Automobile": 0, "Motorbike": 1, "Bike": 2, "Public_Transportation": 3, "Walking": 4}

# Kelas output
classes = [
    'Insufficient_Weight',
    'Normal_Weight',
    'Overweight_Level_I',
    'Overweight_Level_II',
    'Obesity_Type_I',
    'Obesity_Type_II',
    'Obesity_Type_III'
]

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Obesity Predictor", layout="wide")
st.title("🩺 Obesity Level Prediction App")

st.sidebar.header("Masukkan Parameter 👇")
Gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
Age = st.sidebar.slider("Age", 10, 80, 25)
Height = st.sidebar.number_input("Height (m)", 1.2, 2.2, 1.7, 0.01)
Weight = st.sidebar.number_input("Weight (kg)", 30.0, 200.0, 70.0, 0.1)
FHWO = st.sidebar.selectbox("Family History of Overweight?", ["yes", "no"])
FAVC = st.sidebar.selectbox("High caloric food frequently?", ["yes", "no"])
FCVC = st.sidebar.slider("Vegetables consumption (1-3)", 1.0, 3.0, 2.0, 0.1)
NCP = st.sidebar.slider("Main meals per day", 1, 5, 3)
CAEC = st.sidebar.selectbox("Food between meals", ["no", "Sometimes", "Frequently", "Always"])
SMOKE = st.sidebar.selectbox("Do you smoke?", ["yes", "no"])
CH2O = st.sidebar.slider("Water (liters/day)", 1.0, 3.0, 2.0, 0.1)
SCC = st.sidebar.selectbox("Monitor calories?", ["yes", "no"])
FAF = st.sidebar.slider("Physical activity (hours/week)", 0.0, 4.0, 1.0, 0.1)
TUE = st.sidebar.slider("Tech use (hours/day)", 0.0, 16.0, 4.0, 0.5)
CALC = st.sidebar.selectbox("Alcohol consumption", ["no", "Sometimes", "Frequently", "Always"])
MTRANS = st.sidebar.selectbox("Transportation", ["Automobile", "Motorbike", "Bike", "Public_Transportation", "Walking"])

# --------------------------
# Preprocess Input
# --------------------------
input_data = pd.DataFrame([[
    gender_map[Gender],
    Age,
    Height,
    Weight,
    yn_map[FHWO],
    favc_map[FAVC],
    FCVC,
    NCP,
    caec_map[CAEC],
    yn_map[SMOKE],
    CH2O,
    yn_map[SCC],
    FAF,
    TUE,
    calc_map[CALC],
    mtrans_map[MTRANS]
]], columns=["Gender","Age","Height","Weight","FHWO","FAVC","FCVC","NCP","CAEC","SMOKE","CH2O","SCC","FAF","TUE","CALC","MTRANS"])

# --------------------------
# Prediction
# --------------------------
if st.sidebar.button("🔮 Predict Obesity Level"):
    probs = model.predict_proba(input_data)[0]
    pred_class = classes[np.argmax(probs)]
    st.subheader(f"🧠 Hasil Prediksi: **{pred_class}**")

    # Probabilitas semua kelas
    prob_df = pd.DataFrame({
        "Obesity Level": classes,
        "Probability": probs
    }).sort_values("Probability", ascending=False)

    # --------------------------
    # Grafik Probabilitas
    # --------------------------
    fig1 = px.bar(prob_df, x="Obesity Level", y="Probability",
                  title="Probabilitas Prediksi untuk Setiap Level Obesitas",
                  color="Probability", text=prob_df["Probability"].round(2))
    st.plotly_chart(fig1, use_container_width=True)

    # --------------------------
    # Grafik BMI
    # --------------------------
    bmi = Weight / (Height ** 2)
    st.metric("Body Mass Index (BMI)", f"{bmi:.2f}")

    # Buat range kategori BMI
    bmi_categories = {
        "Underweight": (0, 18.5),
        "Normal": (18.5, 24.9),
        "Overweight": (25, 29.9),
        "Obese I": (30, 34.9),
        "Obese II": (35, 39.9),
        "Obese III": (40, 100)
    }
    bmi_labels = list(bmi_categories.keys())
    bmi_values = [np.mean(v) for v in bmi_categories.values()]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=bmi_labels, y=bmi_values, name="BMI Range", marker_color="lightblue"))
    fig2.add_trace(go.Scatter(x=[bmi_labels[np.argmax([low <= bmi <= high for (low, high) in bmi_categories.values()])]],
                              y=[bmi], mode="markers+text", name="Your BMI",
                              text=[f"{bmi:.2f}"], textposition="top center", marker=dict(size=15, color="red")))
    fig2.update_layout(title="BMI Anda vs Kategori BMI")
    st.plotly_chart(fig2, use_container_width=True)

    # --------------------------
    # Radar Chart untuk gaya hidup
    # --------------------------
    radar_labels = ["Vegetables (FCVC)", "Meals/day (NCP)", "Water (CH2O)", "Physical Activity (FAF)", "Tech Use (TUE)"]
    radar_values = [FCVC/3, NCP/5, CH2O/3, FAF/4, TUE/16]  # Normalisasi ke 0-1

    fig3 = go.Figure(data=go.Scatterpolar(
        r=radar_values + [radar_values[0]],  # tutup lingkaran
        theta=radar_labels + [radar_labels[0]],
        fill='toself'
    ))
    fig3.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Radar Chart: Gaya Hidup"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # --------------------------
    # Tabel hasil inputan
    # --------------------------
    st.subheader("📊 Data Input")
    st.dataframe(input_data)

else:
    st.info("Isi parameter di samping, lalu klik **Predict Obesity Level** untuk melihat hasil.")

