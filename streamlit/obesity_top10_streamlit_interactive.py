import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# 1️⃣ Load Model
model = xgb.XGBClassifier()
model.load_model("https://github.com/daudrusyadnurdin/final-project-2025/blob/main/models/xgb_obesity.json")  # path relatif

# 2️⃣ Top 10 Features + default values (sesuai dataset)
default_values = {
    "Gender": 0,   # 0=Male, 1=Female
    "Weight": 70,
    "FCVC": 2,
    "FAVC": 1,     # 1=yes, 0=no
    "CAEC": 1,     # encoded 0-3
    "CALC": 1,     # encoded 0-3
    "Height": 1.70,
    "FHWO": 1,     # 1=yes,0=no
    "NCP": 2,
    "MTRANS": 0    # encoded 0-4
}

# 3️⃣ Sidebar Input
st.sidebar.header("Set Feature Values (Top 10)")

def reset_defaults():
    for k,v in default_values.items():
        st.session_state[k] = v

if st.sidebar.button("Reset to Default"):
    reset_defaults()

feature_inputs = {}

# Gender
feature_inputs["Gender"] = st.sidebar.selectbox(
    "Gender", ["Male", "Female"],
    index=st.session_state.get("Gender", default_values["Gender"])
)
feature_inputs["Gender"] = 0 if feature_inputs["Gender"]=="Male" else 1

# Weight
feature_inputs["Weight"] = st.sidebar.slider(
    "Weight (kg)", 39, 173,
    value=st.session_state.get("Weight", default_values["Weight"])
)

# FCVC
feature_inputs["FCVC"] = st.sidebar.slider(
    "FCVC", 1, 3,
    value=st.session_state.get("FCVC", default_values["FCVC"])
)

# FAVC
favc_options = ["yes","no"]
favc_sel = st.sidebar.selectbox("FAVC", favc_options,
                                index=st.session_state.get("FAVC", default_values["FAVC"]))
feature_inputs["FAVC"] = 1 if favc_sel=="yes" else 0

# CAEC
caec_options = ["no","Sometimes","Frequently","Always"]
caec_sel = st.sidebar.selectbox("CAEC", caec_options,
                                index=st.session_state.get("CAEC", default_values["CAEC"]))
feature_inputs["CAEC"] = caec_options.index(caec_sel)

# CALC
calc_options = ["no","Sometimes","Frequently","Always"]
calc_sel = st.sidebar.selectbox("CALC", calc_options,
                                index=st.session_state.get("CALC", default_values["CALC"]))
feature_inputs["CALC"] = calc_options.index(calc_sel)

# Height
feature_inputs["Height"] = st.sidebar.slider(
    "Height (m)", 1.45, 1.98,
    value=st.session_state.get("Height", default_values["Height"]), step=0.01
)

# FHWO
fhwo_sel = st.sidebar.selectbox("FHWO", ["yes","no"],
                                index=st.session_state.get("FHWO", default_values["FHWO"]))
feature_inputs["FHWO"] = 1 if fhwo_sel=="yes" else 0

# NCP
feature_inputs["NCP"] = st.sidebar.slider(
    "NCP", 1, 4,
    value=st.session_state.get("NCP", default_values["NCP"])
)

# MTRANS
mtrans_options = ["Public_Transportation","Walking","Automobile","Motorbike","Bike"]
mtrans_sel = st.sidebar.selectbox("MTRANS", mtrans_options,
                                  index=st.session_state.get("MTRANS", default_values["MTRANS"]))
feature_inputs["MTRANS"] = mtrans_options.index(mtrans_sel)

# Update session state
for k,v in feature_inputs.items():
    st.session_state[k] = v

# Input DataFrame
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

# 5️⃣ SHAP
st.subheader("Feature Contribution (SHAP Values)")
explainer = shap.TreeExplainer(model)
shap_values = explainer(input_df)

# Plot bar & force side by side
fig, ax = plt.subplots(figsize=(8,5))
shap.plots.bar(shap_values, show=False)
st.pyplot(fig)

st.subheader("Force Plot (Features Driving Prediction)")
shap.initjs()
force_plot_html = shap.force_plot(explainer.expected_value, shap_values.values, input_df, matplotlib=False)
st.components.v1.html(force_plot_html.html(), height=400)


