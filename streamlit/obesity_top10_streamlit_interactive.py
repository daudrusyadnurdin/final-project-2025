import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import requests
import tempfile
import os
import numpy as np

# Set page config
st.set_page_config(page_title="Obesity Prediction", layout="wide")

# Load model dengan error handling yang lebih baik
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/streamlit/xgb-obesity.json"
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        model = xgb.XGBClassifier()
        model.load_model(tmp_path)
        os.unlink(tmp_path)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load model
model = load_model()

if model is None:
    st.stop()

# Debug: Tampilkan fitur yang diharapkan model
st.sidebar.write("🔍 Model features:", model.get_booster().feature_names)

# Default values untuk fitur original (sebelum preprocessing)
default_values = {
    "Gender": "Male",  # Original value
    "Age": 24,
    "Height": 1.70,
    "Weight": 70,
    "FCVC": 2.0,
    "NCP": 3.0,
    "CAEC": "Sometimes",
    "FAVC": "yes",
    "CH2O": 2.0,
    "CALC": "Sometimes",
    "SCC": "no",
    "FAF": 1.0,
    "TUE": 2.0,
    "MTRANS": "Public_Transportation",
    "FHWO": "yes",
    "SMOKE": "no"
}

# Initialize session state
for key in default_values.keys():
    if key not in st.session_state:
        st.session_state[key] = default_values[key]

# 3️⃣ Sidebar Input
st.sidebar.header("🛠️ Set Feature Values")

def reset_defaults():
    for k, v in default_values.items():
        st.session_state[k] = v

if st.sidebar.button("🔄 Reset to Default"):
    reset_defaults()
    st.rerun()

feature_inputs = {}

# Collect user inputs - menggunakan nilai original
# Gender
feature_inputs["Gender"] = st.sidebar.selectbox(
    "Gender", ["Male", "Female"],
    index=0 if st.session_state.get("Gender", default_values["Gender"]) == "Male" else 1
)

# Age
feature_inputs["Age"] = st.sidebar.slider(
    "Age", 14, 61,
    value=st.session_state.get("Age", default_values["Age"])
)

# Height
feature_inputs["Height"] = st.sidebar.slider(
    "Height (m)", 1.45, 1.98,
    value=st.session_state.get("Height", default_values["Height"]), 
    step=0.01
)

# Weight
feature_inputs["Weight"] = st.sidebar.slider(
    "Weight (kg)", 39, 173,
    value=st.session_state.get("Weight", default_values["Weight"])
)

# FCVC - Frequency of consumption of vegetables
feature_inputs["FCVC"] = st.sidebar.slider(
    "FCVC (Vegetable Consumption)", 1.0, 3.0,
    value=float(st.session_state.get("FCVC", default_values["FCVC"])),
    step=0.1
)

# NCP - Number of main meals
feature_inputs["NCP"] = st.sidebar.slider(
    "NCP (Number of Main Meals)", 1.0, 4.0,
    value=float(st.session_state.get("NCP", default_values["NCP"])),
    step=0.1
)

# CAEC - Consumption of food between meals
caec_options = ["no", "Sometimes", "Frequently", "Always"]
feature_inputs["CAEC"] = st.sidebar.selectbox(
    "CAEC (Food Between Meals)", 
    caec_options,
    index=caec_options.index(st.session_state.get("CAEC", default_values["CAEC"]))
)

# FAVC - Frequent consumption of high caloric food
favc_options = ["no", "yes"]
feature_inputs["FAVC"] = st.sidebar.selectbox(
    "FAVC (High Caloric Food)", 
    favc_options,
    index=favc_options.index(st.session_state.get("FAVC", default_values["FAVC"]))
)

# CH2O - Water consumption
feature_inputs["CH2O"] = st.sidebar.slider(
    "CH2O (Water Consumption)", 1.0, 3.0,
    value=float(st.session_state.get("CH2O", default_values["CH2O"])),
    step=0.1
)

# CALC - Consumption of alcohol
calc_options = ["no", "Sometimes", "Frequently", "Always"]
feature_inputs["CALC"] = st.sidebar.selectbox(
    "CALC (Alcohol Consumption)", 
    calc_options,
    index=calc_options.index(st.session_state.get("CALC", default_values["CALC"]))
)

# FAF - Physical activity frequency
feature_inputs["FAF"] = st.sidebar.slider(
    "FAF (Physical Activity)", 0.0, 3.0,
    value=float(st.session_state.get("FAF", default_values["FAF"])),
    step=0.1
)

# TUE - Time using technology devices
feature_inputs["TUE"] = st.sidebar.slider(
    "TUE (Technology Time)", 0.0, 2.0,
    value=float(st.session_state.get("TUE", default_values["TUE"])),
    step=0.1
)

# MTRANS - Transportation used
mtrans_options = ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
feature_inputs["MTRANS"] = st.sidebar.selectbox(
    "MTRANS (Transportation)", 
    mtrans_options,
    index=mtrans_options.index(st.session_state.get("MTRANS", default_values["MTRANS"]))
)

# FHWO - Family history with overweight
fhwo_options = ["no", "yes"]
feature_inputs["FHWO"] = st.sidebar.selectbox(
    "Family History Overweight", 
    fhwo_options,
    index=fhwo_options.index(st.session_state.get("FHWO", default_values["FHWO"]))
)

# SCC - Calories consumption monitoring
scc_options = ["no", "yes"]
feature_inputs["SCC"] = st.sidebar.selectbox(
    "SCC (Calorie Monitoring)", 
    scc_options,
    index=scc_options.index(st.session_state.get("SCC", default_values["SCC"]))
)

# SMOKE - Smoking
smoke_options = ["no", "yes"]
feature_inputs["SMOKE"] = st.sidebar.selectbox(
    "Smoking", 
    smoke_options,
    index=smoke_options.index(st.session_state.get("SMOKE", default_values["SMOKE"]))
)

# Update session state
for k, v in feature_inputs.items():
    st.session_state[k] = v

# Fungsi untuk melakukan preprocessing manual sesuai dengan model
def manual_preprocessing(feature_dict):
    """Manual preprocessing untuk menyesuaikan dengan pipeline model"""
    
    # Create DataFrame dengan fitur original
    original_df = pd.DataFrame([feature_dict])
    
    # ONE-HOT ENCODING untuk categorical features
    processed_features = {}
    
    # Gender: hanya "ohe__Gender_Male" (1 untuk Male, 0 untuk Female)
    processed_features["ohe__Gender_Male"] = [1 if feature_dict["Gender"] == "Male" else 0]
    
    # MTRANS: one-hot encoding untuk 5 kategori
    mtrans_categories = ["Bike", "Motorbike", "Public_Transportation", "Walking", "Automobile"]
    for category in mtrans_categories:
        col_name = f"ohe__MTRANS_{category}"
        processed_features[col_name] = [1 if feature_dict["MTRANS"] == category else 0]
    
    # Numerical features dengan prefix "remainder__"
    numerical_features = ["Age", "Height", "Weight", "FHWO", "FAVC", "FCVC", "NCP", 
                         "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC"]
    
    # Mapping categorical to numerical untuk beberapa fitur
    categorical_to_numerical = {
        "FHWO": {"no": 0, "yes": 1},
        "FAVC": {"no": 0, "yes": 1},
        "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
        "SMOKE": {"no": 0, "yes": 1},
        "SCC": {"no": 0, "yes": 1},
        "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    }
    
    for feature in numerical_features:
        col_name = f"remainder__{feature}"
        if feature in categorical_to_numerical:
            # Convert categorical to numerical
            value = feature_dict[feature]
            processed_features[col_name] = [categorical_to_numerical[feature][value]]
        else:
            # Use numerical value directly
            processed_features[col_name] = [feature_dict[feature]]
    
    # Create final DataFrame dengan urutan yang sesuai model
    expected_features = model.get_booster().feature_names
    final_df = pd.DataFrame(columns=expected_features)
    
    for feature in expected_features:
        if feature in processed_features:
            final_df[feature] = processed_features[feature]
        else:
            final_df[feature] = [0]  # Default value
    
    return final_df

# Create input DataFrame dengan preprocessing manual
input_df = manual_preprocessing(feature_inputs)

# Tampilkan input features sebelum dan sesudah preprocessing
st.header("📊 Input Features")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Features")
    original_df = pd.DataFrame([feature_inputs])
    st.dataframe(original_df)

with col2:
    st.subheader("Preprocessed Features")
    st.dataframe(input_df)

# 4️⃣ Prediction
try:
    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    
    # Map class numbers ke labels yang meaningful
    class_mapping = {
        0: "Insufficient Weight",
        1: "Normal Weight", 
        2: "Overweight Level I",
        3: "Overweight Level II",
        4: "Obesity Level I",
        5: "Obesity Level II",
        6: "Obesity Level III"
    }
    
    predicted_label = class_mapping.get(pred_class, f"Class {pred_class}")
    
    st.header("🎯 Prediction Results")
    
    # Tampilkan hasil prediksi
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Class")
        # Color code based on obesity level
        if "Obesity" in predicted_label:
            st.error(f"**{predicted_label}** ⚠️")
        elif "Overweight" in predicted_label:
            st.warning(f"**{predicted_label}** 📊")
        else:
            st.success(f"**{predicted_label}** ✅")
    
    with col2:
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            'Class': [class_mapping.get(i, f'Class {i}') for i in range(len(pred_proba))],
            'Probability': pred_proba
        }).sort_values('Probability', ascending=False)
        
        # Format probabilities as percentages
        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
        st.dataframe(prob_df, hide_index=True)

    # 5️⃣ SHAP Analysis
    st.header("🔍 SHAP Analysis")
    
    # Explain the prediction
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)
    
    # SHAP Summary Plot - Bar
    st.subheader("Feature Importance (Global)")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    plt.title("Feature Importance - Global Impact")
    plt.tight_layout()
    st.pyplot(fig)
    
    # SHAP Waterfall Plot untuk instance spesifik
    st.subheader("Waterfall Plot - Prediction Explanation")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create Explanation object untuk waterfall plot
        explanation = shap.Explanation(
            values=shap_values.values[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=input_df.columns
        )
        
        shap.waterfall_plot(explanation, show=False)
        plt.title(f"Waterfall Plot for {predicted_label} Prediction")
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Could not display waterfall plot: {e}")
    
    # SHAP Force Plot
    st.subheader("Force Plot - Prediction Forces")
    try:
        plt.figure(figsize=(12, 3))
        force_plot = shap.force_plot(
            explainer.expected_value,
            shap_values.values[0],
            input_df.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning(f"Could not display force plot: {e}")
    
    # Decision Plot
    st.subheader("Decision Plot - Prediction Path")
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.decision_plot(
            explainer.expected_value,
            shap_values.values[0],
            input_df.iloc[0],
            feature_names=list(input_df.columns),
            show=False
        )
        plt.title(f"Decision Plot for {predicted_label}")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display decision plot: {e}")

except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    
    # Detailed debug information
    with st.expander("🔧 Debug Information"):
        st.write("Model classes:", getattr(model, 'classes_', 'Not available'))
        st.write("Input DataFrame shape:", input_df.shape)
        st.write("Input DataFrame columns:", input_df.columns.tolist())
        st.write("Expected features:", model.get_booster().feature_names)
        st.write("Feature inputs:", feature_inputs)
        
        # Check for missing features
        expected = set(model.get_booster().feature_names)
        provided = set(input_df.columns)
        st.write("Missing features:", expected - provided)
        st.write("Extra features:", provided - expected)

# Additional information
st.sidebar.header("ℹ️ About")
st.sidebar.info("""
This model uses preprocessed features with:
- OneHot Encoding for categorical variables
- Standard Scaling for numerical variables
- XGBoost for classification
""")
