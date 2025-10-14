import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# 1️⃣ Load Model
model = xgb.XGBClassifier()
model.load_model("../models/xgb_obesity.json")  # path relatif ke folder models/

# 2️⃣ Top 10 Features
top10_features = ['Gender', 'Weight', 'FCVC', 'FAVC', 'CAEC', 'CALC',
                  'Height', 'FHWO', 'NCP', 'MTRANS']

# 3️⃣ Sidebar Input
st.sidebar.header("Set Feature Values (Top 10)")

feature_inputs = {}

# Gender
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
feature_inputs["Gender"] = 0 if gender == "Male" else 1

# Weight
feature_inputs["Weight"] = st.sidebar.slider("Weight (kg)", 39, 173, 70)

# FCVC
feature_inputs["FCVC"] = st.sidebar.slider("FCVC", 1, 3, 2)

# FAVC
favc = st.sidebar.selectbox("FAVC", ["yes", "no"])
feature_inputs["FAVC"] = 1 if favc == "yes" else 0

# CAEC
caec_options = ["no", "Sometimes", "Frequently", "Always"]
caec = st.sidebar.selectbox("CAEC", caec_options)
feature_inputs["CAEC"] = caec_options.index(caec)

# CALC
calc_options = ["no", "Sometimes", "Frequently", "Always"]
calc = st.sidebar.selectbox("CALC", calc_options)
feature_inputs["CALC"] = calc_options.index(calc)

# Height
feature_inputs["Height"] = st.sidebar.slider("Height (m)", 1.45, 1.98, 1.70, 0.01)

# FHWO
fhwo = st.sidebar.selectbox("FHWO", ["yes", "no"])
feature_inputs["FHWO"] = 1 if fhwo == "yes" else 0

# NCP
feature_inputs["NCP"] = st.sidebar.slider("NCP", 1, 4, 2)

# MTRANS
mtrans_options = ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
mtrans = st.sidebar.selectbox("MTRANS", mtrans_options)
feature_inputs["MTRANS"] = mtrans_options.index(mtrans)

# DataFrame input
input_df = pd.DataFrame(feature_inputs, index=[0])
st.subheader("Input Features")
st.dataframe(input_df)

# 4️⃣ Prediction
pred_class = model.predict(input_df)[0]
pred_proba = model.predict_proba(input_df)

st.subheader("Predicted Obesity Level")
st.write(f"### {pred_class}")
st.subheader("Probability per Class")
st.dataframe(pd.DataFrame(pred_proba, columns=model.classes_))

# 5️⃣ SHAP Feature Importance
st.subheader("Feature Contribution (SHAP Values)")
explainer = shap.TreeExplainer(model)
shap_values = explainer(input_df)

fig, ax = plt.subplots(figsize=(8,5))
shap.plots.bar(shap_values, show=False)
st.pyplot(fig)

# 6️⃣ SHAP Force Plot
st.subheader("Force Plot (Features Driving Prediction)")
shap.initjs()
force_plot_html = shap.force_plot(explainer.expected_value, shap_values.values, input_df, matplotlib=False)
st.components.v1.html(force_plot_html.html(), height=400)
