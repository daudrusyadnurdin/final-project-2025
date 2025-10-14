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
st.set_page_config(
    page_title="Obesity Risk Prediction", 
    layout="wide",
    page_icon="🏥"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .normal-weight {
        background-color: #d4edda;
        border: 2px solid #c3e6cb;
    }
    .overweight {
        background-color: #fff3cd;
        border: 2px solid #ffeaa7;
    }
    .obesity {
        background-color: #f8d7da;
        border: 2px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

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
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Header aplikasi
st.markdown('<h1 class="main-header">🏥 Obesity Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("Predict obesity levels based on lifestyle and physical attributes using Machine Learning")

# Load model
model = load_model()

if model is None:
    st.stop()

st.success("✅ Model loaded successfully!")

# Default values untuk fitur original
default_values = {
    "Gender": "Male",
    "Age": 24,
    "Height": 1.70,
    "Weight": 70,
    "FCVC": 2.0,
    "NCP": 3.0,
    "CAEC": "Sometimes",
    "FAVC": "no",
    "CH2O": 2.0,
    "CALC": "Sometimes",
    "SCC": "no",
    "FAF": 1.0,
    "TUE": 2.0,
    "MTRANS": "Public_Transportation",
    "FHWO": "no",
    "SMOKE": "no"
}

# Initialize session state
for key in default_values.keys():
    if key not in st.session_state:
        st.session_state[key] = default_values[key]

# Sidebar untuk input features
st.sidebar.header("🛠️ Feature Configuration")

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

# Calculate BMI
bmi = feature_inputs["Weight"] / (feature_inputs["Height"] ** 2)
st.sidebar.metric("BMI", f"{bmi:.1f}")

feature_inputs["FCVC"] = st.sidebar.slider(
    "Frequency of vegetable consumption", 1.0, 3.0,
    value=float(st.session_state.get("FCVC", default_values["FCVC"])),
    step=0.1
)

feature_inputs["NCP"] = st.sidebar.slider(
    "Number of main meals per day", 1.0, 4.0,
    value=float(st.session_state.get("NCP", default_values["NCP"])),
    step=0.1
)

caec_options = ["no", "Sometimes", "Frequently", "Always"]
feature_inputs["CAEC"] = st.sidebar.selectbox(
    "Consumption of food between meals", 
    caec_options,
    index=caec_options.index(st.session_state.get("CAEC", default_values["CAEC"]))
)

favc_options = ["no", "yes"]
feature_inputs["FAVC"] = st.sidebar.selectbox(
    "Frequent consumption of high caloric food", 
    favc_options,
    index=favc_options.index(st.session_state.get("FAVC", default_values["FAVC"]))
)

feature_inputs["CH2O"] = st.sidebar.slider(
    "Water consumption (liters per day)", 1.0, 3.0,
    value=float(st.session_state.get("CH2O", default_values["CH2O"])),
    step=0.1
)

calc_options = ["no", "Sometimes", "Frequently", "Always"]
feature_inputs["CALC"] = st.sidebar.selectbox(
    "Alcohol consumption", 
    calc_options,
    index=calc_options.index(st.session_state.get("CALC", default_values["CALC"]))
)

feature_inputs["FAF"] = st.sidebar.slider(
    "Physical activity frequency", 0.0, 3.0,
    value=float(st.session_state.get("FAF", default_values["FAF"])),
    step=0.1
)

feature_inputs["TUE"] = st.sidebar.slider(
    "Time using electronic devices", 0.0, 2.0,
    value=float(st.session_state.get("TUE", default_values["TUE"])),
    step=0.1
)

mtrans_options = ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
feature_inputs["MTRANS"] = st.sidebar.selectbox(
    "Primary transportation method", 
    mtrans_options,
    index=mtrans_options.index(st.session_state.get("MTRANS", default_values["MTRANS"]))
)

scc_options = ["no", "yes"]
feature_inputs["SCC"] = st.sidebar.selectbox(
    "Monitor calorie consumption", 
    scc_options,
    index=scc_options.index(st.session_state.get("SCC", default_values["SCC"]))
)

fhwo_options = ["no", "yes"]
feature_inputs["FHWO"] = st.sidebar.selectbox(
    "Family history of overweight", 
    fhwo_options,
    index=fhwo_options.index(st.session_state.get("FHWO", default_values["FHWO"]))
)

smoke_options = ["no", "yes"]
feature_inputs["SMOKE"] = st.sidebar.selectbox(
    "Smoking habit", 
    smoke_options,
    index=smoke_options.index(st.session_state.get("SMOKE", default_values["SMOKE"]))
)

# Update session state
for k, v in feature_inputs.items():
    st.session_state[k] = v

# Fungsi untuk melakukan preprocessing manual
def manual_preprocessing(feature_dict):
    """Manual preprocessing untuk menyesuaikan dengan pipeline model"""
    
    processed_features = {}
    
    # ONE-HOT ENCODING untuk categorical features
    processed_features["ohe__Gender_Male"] = [1 if feature_dict["Gender"] == "Male" else 0]
    
    # MTRANS: one-hot encoding untuk 5 kategori
    mtrans_categories = ["Bike", "Motorbike", "Public_Transportation", "Walking", "Automobile"]
    for category in mtrans_categories:
        col_name = f"ohe__MTRANS_{category}"
        processed_features[col_name] = [1 if feature_dict["MTRANS"] == category else 0]
    
    # Numerical features dengan prefix "remainder__"
    numerical_features = ["Age", "Height", "Weight", "FHWO", "FAVC", "FCVC", "NCP", 
                         "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC"]
    
    # Mapping categorical to numerical
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

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🔍 Analysis", "ℹ️ About"])

with tab1:
    st.header("📈 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Input Features")
        original_df = pd.DataFrame([feature_inputs])
        st.dataframe(original_df, use_container_width=True)
        
        st.metric("Body Mass Index (BMI)", f"{bmi:.1f}")
        if bmi < 18.5:
            st.info("BMI Category: Underweight")
        elif bmi < 25:
            st.success("BMI Category: Normal weight")
        elif bmi < 30:
            st.warning("BMI Category: Overweight")
        else:
            st.error("BMI Category: Obesity")
    
    with col2:
        st.subheader("Preprocessed Features")
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
    
    # Prediction button
    if st.button("🎯 Predict Obesity Level", type="primary", use_container_width=True):
        with st.spinner("Analyzing features and making prediction..."):
            try:
                pred_class = model.predict(input_df)[0]
                pred_proba = model.predict_proba(input_df)[0]
                
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
                
                st.subheader("🎯 Prediction Result")
                
                if "Obesity" in predicted_label:
                    prediction_class = "obesity"
                    prediction_icon = "⚠️"
                elif "Overweight" in predicted_label:
                    prediction_class = "overweight"
                    prediction_icon = "📊"
                elif "Insufficient" in predicted_label:
                    prediction_class = "normal-weight"
                    prediction_icon = "💪"
                else:
                    prediction_class = "normal-weight"
                    prediction_icon = "✅"
                
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h2 style="text-align: center;">
                        {prediction_icon} {predicted_label} {prediction_icon}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("📊 Class Probabilities")
                
                prob_df = pd.DataFrame({
                    'Obesity Level': [class_mapping.get(i, f'Class {i}') for i in range(len(pred_proba))],
                    'Probability': pred_proba
                }).sort_values('Probability', ascending=False)
                
                prob_df['Probability (%)'] = (prob_df['Probability'] * 100).round(2)
                prob_df = prob_df.drop('Probability', axis=1)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#ff6b6b' if 'Obesity' in x else '#ffd166' if 'Overweight' in x else '#06d6a0' for x in prob_df['Obesity Level']]
                bars = ax.barh(prob_df['Obesity Level'], prob_df['Probability (%)'], color=colors)
                ax.set_xlabel('Probability (%)')
                ax.set_title('Prediction Probabilities by Obesity Level')
                ax.bar_label(bars, fmt='%.1f%%')
                plt.tight_layout()
                st.pyplot(fig)
                
                st.session_state.prediction_results = {
                    'pred_class': pred_class,
                    'pred_proba': pred_proba,
                    'predicted_label': predicted_label,
                    'input_df': input_df,
                    'bmi': bmi
                }
                
            except Exception as e:
                st.error(f"❌ Prediction error: {e}")

with tab2:
    st.header("🔍 Model Analysis")
    
    if 'prediction_results' not in st.session_state:
        st.info("👆 Please make a prediction first in the 'Prediction' tab to see the analysis.")
    else:
        pred_class = st.session_state.prediction_results['pred_class']
        pred_proba = st.session_state.prediction_results['pred_proba']
        predicted_label = st.session_state.prediction_results['predicted_label']
        input_df = st.session_state.prediction_results['input_df']
        
        st.subheader("📊 SHAP Feature Analysis")
        
        with st.spinner("Calculating SHAP values..."):
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                
                # Debug info
                with st.expander("🔧 SHAP Debug Info"):
                    st.write(f"SHAP values shape: {shap_values.shape}")
                    st.write(f"Expected features: {len(input_df.columns)}")
                    st.write(f"Number of classes: {len(model.classes_)}")
                
                # Global Feature Importance
                st.subheader("📈 Global Feature Importance")
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
                plt.title("Overall Feature Importance Across All Classes")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Beeswarm Plot
                st.subheader("🐝 Feature Impact Distribution")
                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(shap_values, input_df, show=False)
                plt.title("Feature Impact Distribution (Beeswarm Plot)")
                plt.tight_layout()
                st.pyplot(fig)
                
                # Class-specific Analysis
                st.subheader(f"🎯 Analysis for {predicted_label}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Waterfall Plot - FIXED
                    try:
                        if len(shap_values.shape) == 3:
                            shap_val_single = shap_values[0, :, pred_class]
                            base_value = explainer.expected_value[pred_class]
                            
                            explanation = shap.Explanation(
                                values=shap_val_single,
                                base_values=base_value,
                                data=input_df.iloc[0].values,
                                feature_names=input_df.columns.tolist()
                            )
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            shap.plots.waterfall(explanation, show=False)
                            plt.title(f"Waterfall Plot for {predicted_label}")
                            plt.tight_layout()
                            st.pyplot(fig)
                            
                    except Exception as e:
                        st.warning(f"Could not display waterfall plot: {e}")
                
                with col2:
                    # Feature Importance untuk class spesifik - FIXED
                    try:
                        if len(shap_values.shape) == 3:
                            shap_val_single = shap_values[0, :, pred_class]
                            
                            # VALIDASI PANJANG ARRAY
                            if len(shap_val_single) == len(input_df.columns):
                                feature_imp = pd.DataFrame({
                                    'feature': input_df.columns,
                                    'importance': np.abs(shap_val_single)
                                }).sort_values('importance', ascending=True)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                bars = ax.barh(feature_imp['feature'], feature_imp['importance'])
                                ax.set_title(f"Feature Importance for {predicted_label}")
                                ax.set_xlabel("Absolute SHAP Value")
                                
                                for bar in bars:
                                    width = bar.get_width()
                                    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                                           f'{width:.3f}', ha='left', va='center')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.error(f"Array length mismatch: SHAP values {len(shap_val_single)} vs features {len(input_df.columns)}")
                        else:
                            st.warning("Unexpected SHAP values shape")
                            
                    except Exception as e:
                        st.warning(f"Could not display feature importance: {e}")
                
                # Decision Plot - FIXED untuk multi-class
                st.subheader(f"🛣️ Decision Path for {predicted_label}")
                try:
                    if len(shap_values.shape) == 3:
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Gunakan values yang benar untuk class yang diprediksi
                        shap_val_class = shap_values.values[0, :, pred_class] if hasattr(shap_values, 'values') else shap_values[0, :, pred_class]
                        
                        shap.decision_plot(
                            explainer.expected_value[pred_class],
                            shap_val_class,
                            feature_names=list(input_df.columns),
                            show=False
                        )
                        plt.title(f"Decision Plot - {predicted_label}")
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.warning(f"Could not display decision plot: {e}")
                
                # SHAP Values Table - FIXED TRANSPOSE
                st.subheader("📋 SHAP Values Table")
                try:
                    if len(shap_values.shape) == 3:
                        # Dapatkan SHAP values untuk semua classes
                        if hasattr(shap_values, 'values'):
                            shap_array = shap_values.values[0]  # Shape: (19, 7)
                        else:
                            shap_array = shap_values[0]  # Shape: (19, 7)
                        
                        # PERBAIKAN: Transpose yang benar
                        # shap_array shape: (features, classes) -> kita mau (classes, features) untuk DataFrame
                        shap_df = pd.DataFrame(
                            shap_array.T,  # Transpose ke (7, 19)
                            columns=input_df.columns,
                            index=[f"Class {i}" for i in range(shap_array.shape[1])]
                        )
                        
                        # Highlight predicted class
                        def highlight_predicted_class(row):
                            if row.name == f"Class {pred_class}":
                                return ['background-color: yellow'] * len(row)
                            return [''] * len(row)
                        
                        styled_df = shap_df.style.apply(highlight_predicted_class, axis=1)\
                                                .background_gradient(cmap='RdBu', axis=None)\
                                                .format("{:.4f}")
                        
                        st.dataframe(styled_df, use_container_width=True)
                        
                        st.caption(f"🟨 Highlighted: {predicted_label} (Class {pred_class})")
                        
                    else:
                        st.warning("Cannot display SHAP values table - unexpected shape")
                        
                except Exception as e:
                    st.warning(f"Could not display SHAP values table: {e}")
                    
                    # Alternative: Simple display tanpa styling
                    try:
                        if len(shap_values.shape) == 3:
                            if hasattr(shap_values, 'values'):
                                shap_array = shap_values.values[0]
                            else:
                                shap_array = shap_values[0]
                            
                            shap_df_simple = pd.DataFrame(
                                shap_array.T,
                                columns=input_df.columns,
                                index=[f"Class {i}" for i in range(shap_array.shape[1])]
                            )
                            st.dataframe(shap_df_simple, use_container_width=True)
                    except:
                        pass
                
                # Additional: Feature Contributions Plot
                st.subheader("📊 Feature Contributions")
                try:
                    if len(shap_values.shape) == 3:
                        shap_val_single = shap_values[0, :, pred_class]
                        
                        # Pisahkan positive dan negative contributions
                        contributions = pd.DataFrame({
                            'feature': input_df.columns,
                            'contribution': shap_val_single,
                            'abs_contribution': np.abs(shap_val_single)
                        }).sort_values('abs_contribution', ascending=True)
                        
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        colors = ['red' if x > 0 else 'green' for x in contributions['contribution']]
                        bars = ax.barh(contributions['feature'], contributions['contribution'], color=colors)
                        
                        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                        ax.set_xlabel('SHAP Value Contribution')
                        ax.set_title(f'Feature Contributions to {predicted_label}\n(Red: Increases risk, Green: Decreases risk)')
                        
                        # Add value labels
                        for bar in bars:
                            width = bar.get_width()
                            if abs(width) > 0.001:  # Only label significant values
                                ax.text(width + (0.01 if width > 0 else -0.01), 
                                       bar.get_y() + bar.get_height()/2, 
                                       f'{width:.3f}', 
                                       ha='left' if width > 0 else 'right', 
                                       va='center')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.warning(f"Could not display feature contributions: {e}")
                    
            except Exception as e:
                st.error(f"❌ SHAP analysis error: {e}")

with tab3:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🏥 Obesity Risk Prediction System
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🏥 Obesity Prediction System | Made with Streamlit & XGBoost"
    "</div>",
    unsafe_allow_html=True
)

# Test examples
st.sidebar.header("🧪 Test Examples")

example_inputs = {
    "Select an example...": default_values,
    "Healthy Lifestyle": {
        "Gender": "Male", "Age": 28, "Height": 1.78, "Weight": 72,
        "FCVC": 2.8, "NCP": 3.0, "CAEC": "Sometimes", "FAVC": "no",
        "CH2O": 2.5, "CALC": "Sometimes", "SCC": "yes", "FAF": 2.5,
        "TUE": 1.0, "MTRANS": "Walking", "FHWO": "no", "SMOKE": "no"
    },
    "Obesity Risk": {
        "Gender": "Female", "Age": 42, "Height": 1.62, "Weight": 88,
        "FCVC": 1.2, "NCP": 2.0, "CAEC": "Frequently", "FAVC": "yes",
        "CH2O": 1.5, "CALC": "no", "SCC": "no", "FAF": 0.5,
        "TUE": 1.8, "MTRANS": "Automobile", "FHWO": "yes", "SMOKE": "no"
    }
}

selected_example = st.sidebar.selectbox("Load test example:", list(example_inputs.keys()))
if st.sidebar.button("🚀 Load Example"):
    if selected_example != "Select an example...":
        example_data = example_inputs[selected_example]
        for k, v in example_data.items():
            st.session_state[k] = v
        st.rerun()
