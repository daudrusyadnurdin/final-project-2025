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
    .feature-importance-bar {
        background: linear-gradient(90deg, #4CAF50, #FFC107, #FF5722);
        height: 8px;
        border-radius: 4px;
        margin: 5px 0;
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
    "FCVC": 2.0,      # Frequency of consumption of vegetables
    "NCP": 3.0,       # Number of main meals
    "CAEC": "Sometimes",  # Consumption of food between meals
    "FAVC": "no",     # Frequent consumption of high caloric food
    "CH2O": 2.0,      # Water consumption
    "CALC": "Sometimes",  # Consumption of alcohol
    "SCC": "no",      # Calories consumption monitoring
    "FAF": 1.0,       # Physical activity frequency
    "TUE": 2.0,       # Time using technology devices
    "MTRANS": "Public_Transportation",  # Transportation used
    "FHWO": "no",     # Family history with overweight
    "SMOKE": "no"     # Smoking
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

# Input features
feature_inputs = {}

# Personal Information
st.sidebar.subheader("👤 Personal Information")

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

# Dietary Habits
st.sidebar.subheader("🍽️ Dietary Habits")

feature_inputs["FCVC"] = st.sidebar.slider(
    "Frequency of vegetable consumption", 1.0, 3.0,
    value=float(st.session_state.get("FCVC", default_values["FCVC"])),
    step=0.1,
    help="1: Never, 2: Sometimes, 3: Always"
)

feature_inputs["NCP"] = st.sidebar.slider(
    "Number of main meals per day", 1.0, 4.0,
    value=float(st.session_state.get("NCP", default_values["NCP"])),
    step=0.1
)

favc_options = ["no", "yes"]
feature_inputs["FAVC"] = st.sidebar.selectbox(
    "Frequent consumption of high caloric food", 
    favc_options,
    index=favc_options.index(st.session_state.get("FAVC", default_values["FAVC"]))
)

caec_options = ["no", "Sometimes", "Frequently", "Always"]
feature_inputs["CAEC"] = st.sidebar.selectbox(
    "Consumption of food between meals", 
    caec_options,
    index=caec_options.index(st.session_state.get("CAEC", default_values["CAEC"]))
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

# Lifestyle Factors
st.sidebar.subheader("🏃 Lifestyle Factors")

feature_inputs["FAF"] = st.sidebar.slider(
    "Physical activity frequency", 0.0, 3.0,
    value=float(st.session_state.get("FAF", default_values["FAF"])),
    step=0.1,
    help="0: No activity, 1: 1-2 days, 2: 2-4 days, 3: 4-5 days"
)

feature_inputs["TUE"] = st.sidebar.slider(
    "Time using electronic devices", 0.0, 2.0,
    value=float(st.session_state.get("TUE", default_values["TUE"])),
    step=0.1,
    help="0: 0-2 hours, 1: 3-5 hours, 2: More than 5 hours"
)

mtrans_options = ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
feature_inputs["MTRANS"] = st.sidebar.selectbox(
    "Primary transportation method", 
    mtrans_options,
    index=mtrans_options.index(st.session_state.get("MTRANS", default_values["MTRANS"]))
)

# Health Information
st.sidebar.subheader("❤️ Health Information")

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
            final_df[feature] = [0]  # Default value
    
    return final_df

# Create input DataFrame dengan preprocessing manual
input_df = manual_preprocessing(feature_inputs)

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Prediction", "🔍 Analysis", "ℹ️ About"])

with tab1:
    st.header("📈 Prediction Results")
    
    # Tampilkan input features
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Input Features")
        original_df = pd.DataFrame([feature_inputs])
        st.dataframe(original_df, use_container_width=True)
        
        # Display BMI information
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
                # Make prediction
                pred_class = model.predict(input_df)[0]
                pred_proba = model.predict_proba(input_df)[0]
                
                # Map class numbers ke labels
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
                
                # Display prediction results
                st.subheader("🎯 Prediction Result")
                
                # Color-coded prediction box
                if "Obesity" in predicted_label:
                    prediction_class = "obesity"
                    prediction_icon = "⚠️"
                    prediction_color = "red"
                elif "Overweight" in predicted_label:
                    prediction_class = "overweight"
                    prediction_icon = "📊"
                    prediction_color = "orange"
                elif "Insufficient" in predicted_label:
                    prediction_class = "normal-weight"
                    prediction_icon = "💪"
                    prediction_color = "blue"
                else:
                    prediction_class = "normal-weight"
                    prediction_icon = "✅"
                    prediction_color = "green"
                
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h2 style="color: {prediction_color}; text-align: center;">
                        {prediction_icon} {predicted_label} {prediction_icon}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Display probabilities
                st.subheader("📊 Class Probabilities")
                
                prob_df = pd.DataFrame({
                    'Obesity Level': [class_mapping.get(i, f'Class {i}') for i in range(len(pred_proba))],
                    'Probability': pred_proba
                }).sort_values('Probability', ascending=False)
                
                prob_df['Probability (%)'] = (prob_df['Probability'] * 100).round(2)
                prob_df = prob_df.drop('Probability', axis=1)
                
                # Display as bar chart
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['#ff6b6b' if 'Obesity' in x else '#ffd166' if 'Overweight' in x else '#06d6a0' for x in prob_df['Obesity Level']]
                bars = ax.barh(prob_df['Obesity Level'], prob_df['Probability (%)'], color=colors)
                ax.set_xlabel('Probability (%)')
                ax.set_title('Prediction Probabilities by Obesity Level')
                ax.bar_label(bars, fmt='%.1f%%')
                plt.tight_layout()
                st.pyplot(fig)
                
                # Store prediction results in session state for other tabs
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
        bmi = st.session_state.prediction_results.get('bmi', 0)
        
        # SHAP Analysis
        st.subheader("📊 SHAP Feature Analysis")
        
        with st.spinner("Calculating SHAP values..."):
            try:
                # Initialize SHAP explainer
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(input_df)
                
                # Display SHAP values shape info
                with st.expander("📐 SHAP Values Shape Info"):
                    st.write(f"SHAP values shape: {shap_values.shape}")
                    st.write(f"Number of classes: {len(model.classes_)}")
                    st.write(f"Number of features: {len(input_df.columns)}")
                
                # Global Feature Importance - BAR PLOT
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
                    # Waterfall Plot untuk class yang diprediksi - FIXED
                    try:
                        # Untuk multi-class, shape: (1, n_features, n_classes)
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
                        else:
                            st.warning("Unexpected SHAP values shape for waterfall plot")
                            
                    except Exception as e:
                        st.warning(f"Could not display waterfall plot: {e}")
                
                with col2:
                    # Feature Importance untuk class spesifik - FIXED
                    try:
                        if len(shap_values.shape) == 3:
                            # Ambil SHAP values untuk class yang diprediksi
                            shap_val_single = shap_values[0, :, pred_class]
                            
                            # Pastikan panjang array sesuai
                            if len(shap_val_single) == len(input_df.columns):
                                feature_imp = pd.DataFrame({
                                    'feature': input_df.columns,
                                    'importance': np.abs(shap_val_single)
                                }).sort_values('importance', ascending=True)
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                bars = ax.barh(feature_imp['feature'], feature_imp['importance'])
                                ax.set_title(f"Feature Importance for {predicted_label}")
                                ax.set_xlabel("Absolute SHAP Value")
                                
                                # Tambah nilai pada bar
                                for bar in bars:
                                    width = bar.get_width()
                                    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                                           f'{width:.3f}', ha='left', va='center')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            else:
                                st.warning("Feature count mismatch in SHAP values")
                        else:
                            st.warning("Unexpected SHAP values shape")
                            
                    except Exception as e:
                        st.warning(f"Could not display feature importance: {e}")
                
                # Decision Plot untuk class yang diprediksi - FIXED
                st.subheader(f"🛣️ Decision Path for {predicted_label}")
                try:
                    if len(shap_values.shape) == 3:
                        # Untuk multi-output, gunakan base_value[i] dan shap_values[i]
                        fig, ax = plt.subplots(figsize=(12, 8))
                        
                        # Gunakan multioutput_decision_plot atau decision_plot untuk class spesifik
                        shap.decision_plot(
                            explainer.expected_value[pred_class],
                            shap_values.values[0, :, pred_class] if hasattr(shap_values, 'values') else shap_values[0, :, pred_class],
                            feature_names=list(input_df.columns),
                            show=False
                        )
                        plt.title(f"Decision Plot - {predicted_label}")
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.warning("Multi-output decision plot requires specific class selection")
                        
                except Exception as e:
                    st.warning(f"Could not display decision plot: {e}")
                    
                    # Alternative: Simple feature contribution plot
                    try:
                        if len(shap_values.shape) == 3:
                            shap_val_single = shap_values[0, :, pred_class]
                            
                            fig, ax = plt.subplots(figsize=(12, 8))
                            
                            # Plot positive and negative contributions separately
                            positive_idx = shap_val_single > 0
                            negative_idx = shap_val_single < 0
                            
                            features = np.array(input_df.columns)
                            
                            if np.any(positive_idx):
                                ax.barh(features[positive_idx], shap_val_single[positive_idx], 
                                       color='red', label='Increases Obesity Risk')
                            if np.any(negative_idx):
                                ax.barh(features[negative_idx], shap_val_single[negative_idx], 
                                       color='green', label='Decreases Obesity Risk')
                            
                            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                            ax.set_xlabel('SHAP Value Impact')
                            ax.set_title(f'Feature Contributions for {predicted_label}')
                            ax.legend()
                            plt.tight_layout()
                            st.pyplot(fig)
                    except Exception as e2:
                        st.error(f"Alternative plot also failed: {e2}")
                
                # SHAP Values Table untuk semua classes
                st.subheader("📋 SHAP Values Table")
                try:
                    if len(shap_values.shape) == 3:
                        # Create DataFrame dengan SHAP values untuk semua classes
                        shap_df = pd.DataFrame(
                            shap_values.values[0].T if hasattr(shap_values, 'values') else shap_values[0].T,
                            columns=[f"Class {i}" for i in range(shap_values.shape[2])],
                            index=input_df.columns
                        )
                        
                        # Highlight the predicted class
                        styled_df = shap_df.style.background_gradient(cmap='RdBu', axis=1)\
                                                .format("{:.4f}")\
                                                .set_caption(f"SHAP Values for Each Feature and Class (Predicted: {predicted_label})")
                        
                        st.dataframe(styled_df, use_container_width=True)
                    else:
                        st.warning("Cannot display SHAP values table - unexpected shape")
                        
                except Exception as e:
                    st.warning(f"Could not display SHAP values table: {e}")
                    
            except Exception as e:
                st.error(f"❌ SHAP analysis error: {e}")

with tab3:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🏥 Obesity Risk Prediction System
    
    This machine learning application predicts obesity levels based on lifestyle, dietary habits, 
    and physical attributes using an XGBoost classifier.
    
    #### 📋 Features Used:
    
    **Personal Information:**
    - Gender, Age, Height, Weight
    
    **Dietary Habits:**
    - Vegetable consumption frequency (FCVC)
    - Number of main meals (NCP)
    - High caloric food consumption (FAVC)
    - Food between meals (CAEC)
    - Water consumption (CH2O)
    - Alcohol consumption (CALC)
    
    **Lifestyle Factors:**
    - Physical activity frequency (FAF)
    - Technology usage time (TUE)
    - Transportation method (MTRANS)
    
    **Health Information:**
    - Calorie monitoring (SCC)
    - Family history of overweight (FHWO)
    - Smoking habit (SMOKE)
    
    #### 🎯 Obesity Levels:
    - **Insufficient Weight**: BMI < 18.5
    - **Normal Weight**: BMI 18.5-24.9
    - **Overweight Level I**: BMI 25-26.9
    - **Overweight Level II**: BMI 27-29.9
    - **Obesity Level I**: BMI 30-34.9
    - **Obesity Level II**: BMI 35-39.9
    - **Obesity Level III**: BMI ≥ 40
    
    #### 🔧 Technical Details:
    - **Model**: XGBoost Classifier
    - **Preprocessing**: OneHot Encoding + Standard Scaling
    - **Interpretability**: SHAP (SHapley Additive exPlanations)
    - **Framework**: Streamlit for web interface
    
    #### 📊 Model Performance:
    The model has been trained on comprehensive obesity dataset and provides 
    explainable predictions with feature importance analysis.
    """)
    
    # Feature descriptions
    with st.expander("📖 Detailed Feature Descriptions"):
        st.markdown("""
        | Feature | Description | Values |
        |---------|-------------|--------|
        | **Gender** | Biological sex | Male, Female |
        | **Age** | Age in years | 14-61 |
        | **Height** | Height in meters | 1.45-1.98m |
        | **Weight** | Weight in kilograms | 39-173kg |
        | **FCVC** | Frequency of vegetable consumption | 1-3 (Never-Always) |
        | **NCP** | Number of main meals | 1-4 meals per day |
        | **FAVC** | High caloric food consumption | Yes/No |
        | **CAEC** | Food between meals | No, Sometimes, Frequently, Always |
        | **CH2O** | Water consumption | 1-3 (Low-High) |
        | **CALC** | Alcohol consumption | No, Sometimes, Frequently, Always |
        | **SCC** | Calorie consumption monitoring | Yes/No |
        | **FAF** | Physical activity frequency | 0-3 (None-High) |
        | **TUE** | Technology device usage time | 0-2 (Low-High) |
        | **MTRANS** | Transportation method | Various methods |
        | **FHWO** | Family history of overweight | Yes/No |
        | **SMOKE** | Smoking habit | Yes/No |
        """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🏥 Obesity Prediction System | Made with Streamlit & XGBoost | "
    "For educational and health awareness purposes"
    "</div>",
    unsafe_allow_html=True
)

# Test examples di sidebar
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
    },
    "Underweight": {
        "Gender": "Female", "Age": 22, "Height": 1.68, "Weight": 48,
        "FCVC": 2.0, "NCP": 2.5, "CAEC": "Sometimes", "FAVC": "no",
        "CH2O": 2.0, "CALC": "no", "SCC": "yes", "FAF": 1.5,
        "TUE": 1.2, "MTRANS": "Bike", "FHWO": "no", "SMOKE": "no"
    }
}

selected_example = st.sidebar.selectbox("Load test example:", list(example_inputs.keys()))
if st.sidebar.button("🚀 Load Example", use_container_width=True):
    if selected_example != "Select an example...":
        example_data = example_inputs[selected_example]
        for k, v in example_data.items():
            st.session_state[k] = v
        st.rerun()
