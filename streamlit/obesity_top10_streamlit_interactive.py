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

# Default values untuk fitur original (sebelum preprocessing)
default_values = {
    "Gender": "Male",
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

# Sidebar Input
st.sidebar.header("🛠️ Set Feature Values")

def reset_defaults():
    for k, v in default_values.items():
        st.session_state[k] = v

if st.sidebar.button("🔄 Reset to Default"):
    reset_defaults()
    st.rerun()

feature_inputs = {}

# Collect user inputs
feature_inputs["Gender"] = st.sidebar.selectbox(
    "Gender", ["Male", "Female"],
    index=0 if st.session_state.get("Gender", default_values["Gender"]) == "Male" else 1
)

feature_inputs["Age"] = st.sidebar.slider(
    "Age", 14, 61,
    value=st.session_state.get("Age", default_values["Age"])
)

feature_inputs["Height"] = st.sidebar.slider(
    "Height (m)", 1.45, 1.98,
    value=st.session_state.get("Height", default_values["Height"]), 
    step=0.01
)

feature_inputs["Weight"] = st.sidebar.slider(
    "Weight (kg)", 39, 173,
    value=st.session_state.get("Weight", default_values["Weight"])
)

feature_inputs["FCVC"] = st.sidebar.slider(
    "FCVC (Vegetable Consumption)", 1.0, 3.0,
    value=float(st.session_state.get("FCVC", default_values["FCVC"])),
    step=0.1
)

feature_inputs["NCP"] = st.sidebar.slider(
    "NCP (Number of Main Meals)", 1.0, 4.0,
    value=float(st.session_state.get("NCP", default_values["NCP"])),
    step=0.1
)

caec_options = ["no", "Sometimes", "Frequently", "Always"]
feature_inputs["CAEC"] = st.sidebar.selectbox(
    "CAEC (Food Between Meals)", 
    caec_options,
    index=caec_options.index(st.session_state.get("CAEC", default_values["CAEC"]))
)

favc_options = ["no", "yes"]
feature_inputs["FAVC"] = st.sidebar.selectbox(
    "FAVC (High Caloric Food)", 
    favc_options,
    index=favc_options.index(st.session_state.get("FAVC", default_values["FAVC"]))
)

feature_inputs["CH2O"] = st.sidebar.slider(
    "CH2O (Water Consumption)", 1.0, 3.0,
    value=float(st.session_state.get("CH2O", default_values["CH2O"])),
    step=0.1
)

calc_options = ["no", "Sometimes", "Frequently", "Always"]
feature_inputs["CALC"] = st.sidebar.selectbox(
    "CALC (Alcohol Consumption)", 
    calc_options,
    index=calc_options.index(st.session_state.get("CALC", default_values["CALC"]))
)

feature_inputs["FAF"] = st.sidebar.slider(
    "FAF (Physical Activity)", 0.0, 3.0,
    value=float(st.session_state.get("FAF", default_values["FAF"])),
    step=0.1
)

feature_inputs["TUE"] = st.sidebar.slider(
    "TUE (Technology Time)", 0.0, 2.0,
    value=float(st.session_state.get("TUE", default_values["TUE"])),
    step=0.1
)

mtrans_options = ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
feature_inputs["MTRANS"] = st.sidebar.selectbox(
    "MTRANS (Transportation)", 
    mtrans_options,
    index=mtrans_options.index(st.session_state.get("MTRANS", default_values["MTRANS"]))
)

fhwo_options = ["no", "yes"]
feature_inputs["FHWO"] = st.sidebar.selectbox(
    "Family History Overweight", 
    fhwo_options,
    index=fhwo_options.index(st.session_state.get("FHWO", default_values["FHWO"]))
)

scc_options = ["no", "yes"]
feature_inputs["SCC"] = st.sidebar.selectbox(
    "SCC (Calorie Monitoring)", 
    scc_options,
    index=scc_options.index(st.session_state.get("SCC", default_values["SCC"]))
)

smoke_options = ["no", "yes"]
feature_inputs["SMOKE"] = st.sidebar.selectbox(
    "Smoking", 
    smoke_options,
    index=smoke_options.index(st.session_state.get("SMOKE", default_values["SMOKE"]))
)

# Update session state
for k, v in feature_inputs.items():
    st.session_state[k] = v

# Fungsi untuk melakukan preprocessing manual
def manual_preprocessing(feature_dict):
    """Manual preprocessing untuk menyesuaikan dengan pipeline model"""
    
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
            value = feature_dict[feature]
            processed_features[col_name] = [categorical_to_numerical[feature][value]]
        else:
            processed_features[col_name] = [feature_dict[feature]]
    
    # Create final DataFrame dengan urutan yang sesuai model
    expected_features = model.get_booster().feature_names
    final_df = pd.DataFrame(columns=expected_features)
    
    for feature in expected_features:
        if feature in processed_features:
            final_df[feature] = processed_features[feature]
        else:
            final_df[feature] = [0]
    
    return final_df

# Create input DataFrame dengan preprocessing manual
input_df = manual_preprocessing(feature_inputs)

# Tampilkan input features
st.header("📊 Input Features")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Original Features")
    original_df = pd.DataFrame([feature_inputs])
    st.dataframe(original_df)

with col2:
    st.subheader("Preprocessed Features")
    st.dataframe(input_df)

# Prediction
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
        
        prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
        st.dataframe(prob_df, hide_index=True)

    # SHAP Analysis - FIXED FOR MULTI-CLASS
    st.header("🔍 SHAP Analysis")
    
    # Explain the prediction
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_df)
    
    # SHAP Summary Plot - Bar (Global feature importance)
    st.subheader("Global Feature Importance")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
    plt.title("Overall Feature Importance Across All Classes")
    plt.tight_layout()
    st.pyplot(fig)
    
    # SHAP Summary Plot - Beeswarm
    st.subheader("Beeswarm Plot - Feature Impact Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, input_df, show=False)
    plt.title("Feature Impact Distribution Across Classes")
    plt.tight_layout()
    st.pyplot(fig)
    
    # WATERFALL PLOT FIXED - untuk class yang diprediksi
    st.subheader(f"Waterfall Plot - {predicted_label} Prediction")
    try:
        # Untuk multi-class, kita perlu memilih SHAP values untuk class yang diprediksi
        class_idx = pred_class
        
        # shap_values adalah matrix (1 instance × 19 features × 7 classes)
        # Kita ambil untuk instance pertama dan class yang diprediksi
        if len(shap_values.shape) == 3:
            # Multi-class scenario: shap_values[instance, features, class]
            shap_val_single = shap_values[0, :, class_idx]  # Shape: (19,)
            base_value = explainer.expected_value[class_idx]
        else:
            # Fallback: jika shape berbeda
            shap_val_single = shap_values[0, :]  # Shape: (19,)
            base_value = explainer.expected_value
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_val_single,
            base_values=base_value,
            data=input_df.iloc[0].values,
            feature_names=input_df.columns.tolist()
        )
        
        fig, ax = plt.subplots(figsize=(12, 6))
        shap.waterfall_plot(explanation, show=False)
        plt.title(f"Waterfall Plot for {predicted_label}\n(Base value: {base_value:.4f})")
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.warning(f"Could not display waterfall plot: {e}")
        st.info("Trying alternative approach...")
        
        # Alternative approach
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            shap.plots.waterfall(shap_values[0, :, pred_class], show=False)
            plt.title(f"Waterfall Plot for {predicted_label} (Alternative)")
            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e2:
            st.error(f"Alternative also failed: {e2}")
    
    # Force Plot untuk class yang diprediksi - FIXED VERSION
st.subheader(f"Force Plot - {predicted_label}")
try:
    if len(shap_values.shape) == 3:
        shap_val_single = shap_values[0, :, pred_class]
        base_value = explainer.expected_value[pred_class]
    else:
        shap_val_single = shap_values[0, :]
        base_value = explainer.expected_value
    
    # CREATE EXPLANATION OBJECT FIRST
    explanation = shap.Explanation(
        values=shap_val_single,
        base_values=base_value,
        data=input_df.iloc[0].values,
        feature_names=input_df.columns.tolist()
    )
    
    plt.figure(figsize=(12, 3))
    
    # METHOD 1: Menggunakan explanation object
    shap.plots.force(explanation, matplotlib=True, show=False)
    plt.title(f"Force Plot for {predicted_label}")
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
except Exception as e:
    st.warning(f"Could not display force plot with method 1: {e}")
    
    # METHOD 2: Alternative approach
    try:
        plt.figure(figsize=(12, 3))
        
        # Untuk multi-output models
        if len(shap_values.shape) == 3:
            shap.plots.force(explainer.expected_value[pred_class], 
                           shap_values.values[0, :, pred_class], 
                           input_df.iloc[0], 
                           matplotlib=True, show=False)
        else:
            shap.plots.force(explainer.expected_value, 
                           shap_values.values[0, :], 
                           input_df.iloc[0], 
                           matplotlib=True, show=False)
        
        plt.title(f"Force Plot for {predicted_label} (Method 2)")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        
    except Exception as e2:
        st.warning(f"Could not display force plot with method 2: {e2}")
        
        # METHOD 3: Simple bar plot sebagai alternatif
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Create feature importance plot
            feature_importance = pd.DataFrame({
                'feature': input_df.columns,
                'importance': np.abs(shap_val_single)
            }).sort_values('importance', ascending=True)
            
            ax.barh(feature_importance['feature'], feature_importance['importance'])
            ax.set_title(f"Feature Importance for {predicted_label} Prediction")
            ax.set_xlabel("Absolute SHAP Value")
            plt.tight_layout()
            st.pyplot(fig)
            
        except Exception as e3:
            st.error(f"All force plot methods failed: {e3}")
    
    # Decision Plot untuk semua classes
    st.subheader("Decision Plot - All Classes")
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.decision_plot(
            explainer.expected_value,
            shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0],
            feature_names=list(input_df.columns),
            show=False
        )
        plt.title("Decision Plot - Prediction Path for All Classes")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not display decision plot: {e}")
    
    # Class-specific SHAP values
    st.subheader("SHAP Values per Class")
    try:
        # Create DataFrame dengan SHAP values untuk setiap class
        shap_df = pd.DataFrame(
            shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0],
            columns=[class_mapping.get(i, f'Class {i}') for i in range(shap_values.shape[2])],
            index=input_df.columns
        )
        st.dataframe(shap_df.style.background_gradient(cmap='RdBu', axis=1))
    except Exception as e:
        st.warning(f"Could not display SHAP values table: {e}")

except Exception as e:
    st.error(f"❌ Prediction error: {e}")
    
    with st.expander("🔧 Debug Information"):
        st.write("Model classes:", getattr(model, 'classes_', 'Not available'))
        st.write("Input DataFrame shape:", input_df.shape)
        st.write("Input DataFrame columns:", input_df.columns.tolist())
        st.write("Expected features:", model.get_booster().feature_names)

# Contoh input yang benar untuk testing
st.sidebar.header("🧪 Test Input Examples")

example_inputs = {
    "Example 1 - Normal Weight": {
        "Gender": "Male", "Age": 25, "Height": 1.75, "Weight": 70,
        "FCVC": 2.5, "NCP": 3.0, "CAEC": "Sometimes", "FAVC": "no",
        "CH2O": 2.0, "CALC": "Sometimes", "SCC": "no", "FAF": 2.0,
        "TUE": 1.0, "MTRANS": "Walking", "FHWO": "no", "SMOKE": "no"
    },
    "Example 2 - Obesity Risk": {
        "Gender": "Female", "Age": 45, "Height": 1.60, "Weight": 85,
        "FCVC": 1.0, "NCP": 2.0, "CAEC": "Frequently", "FAVC": "yes",
        "CH2O": 1.0, "CALC": "no", "SCC": "no", "FAF": 0.5,
        "TUE": 2.0, "MTRANS": "Automobile", "FHWO": "yes", "SMOKE": "no"
    }
}

selected_example = st.sidebar.selectbox("Load Example:", list(example_inputs.keys()))
if st.sidebar.button("Apply Example"):
    example_data = example_inputs[selected_example]
    for k, v in example_data.items():
        st.session_state[k] = v
    st.rerun()

