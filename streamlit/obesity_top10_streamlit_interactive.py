import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import requests
import tempfile
import os

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

# Default values sesuai dengan fitur model
default_values = {
    "Gender": 0,   # 0=Male, 1=Female
    "Age": 24,     # Ditambahkan karena mungkin diperlukan
    "Height": 1.70,
    "Weight": 70,
    "FCVC": 2,     # Frequency of consumption of vegetables
    "NCP": 3,      # Number of main meals
    "CAEC": 1,     # Consumption of food between meals
    "FAVC": 1,     # Frequent consumption of high caloric food
    "CH2O": 2,     # Water consumption
    "CALC": 1,     # Consumption of alcohol
    "SCC": 0,      # Calories consumption monitoring
    "FAF": 1,      # Physical activity frequency
    "TUE": 2,      # Time using technology devices
    "MTRANS": 0,   # Transportation used
    "FHWO": 1,     # Family history with overweight
    "SMOKE": 0,    # Smoking
    "family_history_with_overweight": 1  # Mungkin diperlukan
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

# Collect user inputs
# Gender
gender_sel = st.sidebar.selectbox(
    "Gender", ["Male", "Female"],
    index=st.session_state.get("Gender", default_values["Gender"])
)
feature_inputs["Gender"] = 0 if gender_sel == "Male" else 1

# Age (ditambahkan karena biasanya diperlukan)
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
caec_sel = st.sidebar.selectbox(
    "CAEC (Food Between Meals)", 
    caec_options,
    index=st.session_state.get("CAEC", default_values["CAEC"])
)
feature_inputs["CAEC"] = caec_options.index(caec_sel)

# FAVC - Frequent consumption of high caloric food
favc_options = ["no", "yes"]
favc_sel = st.sidebar.selectbox(
    "FAVC (High Caloric Food)", 
    favc_options,
    index=st.session_state.get("FAVC", default_values["FAVC"])
)
feature_inputs["FAVC"] = favc_options.index(favc_sel)

# CH2O - Water consumption (ditambahkan)
feature_inputs["CH2O"] = st.sidebar.slider(
    "CH2O (Water Consumption)", 1.0, 3.0,
    value=float(st.session_state.get("CH2O", default_values["CH2O"])),
    step=0.1
)

# CALC - Consumption of alcohol
calc_options = ["no", "Sometimes", "Frequently", "Always"]
calc_sel = st.sidebar.selectbox(
    "CALC (Alcohol Consumption)", 
    calc_options,
    index=st.session_state.get("CALC", default_values["CALC"])
)
feature_inputs["CALC"] = calc_options.index(calc_sel)

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
mtrans_sel = st.sidebar.selectbox(
    "MTRANS (Transportation)", 
    mtrans_options,
    index=st.session_state.get("MTRANS", default_values["MTRANS"])
)
feature_inputs["MTRANS"] = mtrans_options.index(mtrans_sel)

# FHWO - Family history with overweight
fhwo_options = ["no", "yes"]
fhwo_sel = st.sidebar.selectbox(
    "Family History Overweight", 
    fhwo_options,
    index=st.session_state.get("FHWO", default_values["FHWO"])
)
feature_inputs["FHWO"] = fhwo_options.index(fhwo_sel)
feature_inputs["family_history_with_overweight"] = feature_inputs["FHWO"]  # Duplicate untuk kompatibilitas

# SCC - Calories consumption monitoring
scc_options = ["no", "yes"]
scc_sel = st.sidebar.selectbox(
    "SCC (Calorie Monitoring)", 
    scc_options,
    index=st.session_state.get("SCC", default_values["SCC"])
)
feature_inputs["SCC"] = scc_options.index(scc_sel)

# SMOKE - Smoking
smoke_options = ["no", "yes"]
smoke_sel = st.sidebar.selectbox(
    "Smoking", 
    smoke_options,
    index=st.session_state.get("SMOKE", default_values["SMOKE"])
)
feature_inputs["SMOKE"] = smoke_options.index(smoke_sel)

# Update session state
for k, v in feature_inputs.items():
    st.session_state[k] = v

# Prepare input DataFrame dengan urutan yang benar
def prepare_input_data(feature_dict, model):
    """Prepare input data dengan urutan fitur yang sesuai dengan model"""
    expected_features = model.get_booster().feature_names
    
    # Create DataFrame dengan urutan yang benar
    input_data = {}
    for feature in expected_features:
        if feature in feature_dict:
            input_data[feature] = [feature_dict[feature]]
        else:
            # Jika fitur tidak ada, gunakan default value
            input_data[feature] = [default_values.get(feature, 0)]
    
    return pd.DataFrame(input_data, columns=expected_features)

# Create input DataFrame
input_df = prepare_input_data(feature_inputs, model)

# Tampilkan input features
st.header("📊 Input Features")
st.dataframe(input_df)

# 4️⃣ Prediction
try:
    pred_class = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0]
    
    # Map class numbers ke labels yang meaningful
    class_mapping = {
        0: "Normal Weight",
        1: "Overweight Level I", 
        2: "Overweight Level II",
        3: "Obesity Level I",
        4: "Obesity Level II",
        5: "Obesity Level III"
    }
    
    predicted_label = class_mapping.get(pred_class, f"Class {pred_class}")
    
    st.header("🎯 Prediction Results")
    
    # Tampilkan hasil prediksi
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Predicted Class")
        st.info(f"**{predicted_label}**")
    
    with col2:
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({
            'Class': [class_mapping.get(i, f'Class {i}') for i in range(len(pred_proba))],
            'Probability': pred_proba
        })
        st.dataframe(prob_df)
        
        # Highlight highest probability
        max_prob_idx = pred_proba.argmax()
        st.write(f"Highest probability: **{class_mapping.get(max_prob_idx, f'Class {max_prob_idx}')}** ({pred_proba[max_prob_idx]:.2%})")

    # 5️⃣ SHAP Analysis - FIXED VERSION
    st.header("🔍 SHAP Analysis")
    
    # Explain the prediction
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)
    
    # SHAP Summary Plot
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    plt.tight_layout()
    st.pyplot(fig)
    
    # SHAP Force Plot - FIXED SYNTAX
    st.subheader("Force Plot - Prediction Explanation")
    
    # Untuk single instance prediction
    if hasattr(shap_values, 'values'):
        # New SHAP version syntax
        force_fig = shap.force_plot(
            explainer.expected_value,
            shap_values.values[0],  # First instance
            input_df.iloc[0],
            matplotlib=True,
            show=False
        )
        st.pyplot(force_fig)
    else:
        # Alternative approach
        try:
            # Coba approach yang berbeda untuk SHAP versions
            fig, ax = plt.subplots(figsize=(12, 4))
            shap.decision_plot(explainer.expected_value, 
                             shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0],
                             input_df.iloc[0], 
                             show=False)
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.warning(f"Could not display force plot: {e}")
    
    # Waterfall plot untuk penjelasan detail
    st.subheader("Waterfall Plot - Detailed Feature Contributions")
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Untuk SHAP versions yang berbeda
        if hasattr(shap_values, 'values'):
            shap.waterfall_plot(shap.Explanation(
                values=shap_values.values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            ), show=False)
        else:
            shap.waterfall_plot(shap.Explanation(
                values=shap_values[0],
                base_values=explainer.expected_value,
                data=input_df.iloc[0],
                feature_names=input_df.columns
            ), show=False)
            
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display waterfall plot: {e}")
    
    # Alternative: Beeswarm plot untuk overview
    st.subheader("Beeswarm Plot - Overall Feature Impact")
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, input_df, show=False)
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display beeswarm plot: {e}")

except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    st.info("Please check that all required features are provided correctly")
    
    # Debug information
    with st.expander("Debug Information"):
        st.write("Model classes:", model.classes_)
        st.write("Input DataFrame shape:", input_df.shape)
        st.write("Input DataFrame columns:", input_df.columns.tolist())
        st.write("Expected features:", model.get_booster().feature_names)
