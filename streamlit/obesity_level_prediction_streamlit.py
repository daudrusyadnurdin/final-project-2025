import streamlit as st
import pandas as pd
import xgboost as xgb
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
        text-align: center;
    }
    .insufficient-weight {
        background-color: #e3f2fd;
        border: 2px solid #90caf9;
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
        
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_file.write(response.text)
            tmp_path = tmp_file.name
        
        # Load model
        model = xgb.Booster()
        model.load_model(tmp_path)
        os.unlink(tmp_path)
        
        st.success("✅ Model loaded successfully!")
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
    st.error("Failed to load model. Please check the model URL.")
    st.stop()

# Default values
default_values = {
    "Gender": "Male",
    "Age": 24.0,
    "Height": 1.70,
    "Weight": 70.0,
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

# Collect user inputs
feature_inputs = {}

feature_inputs["Gender"] = st.sidebar.selectbox(
    "Gender", ["Male", "Female"],
    index=0 if st.session_state.get("Gender", default_values["Gender"]) == "Male" else 1
)

feature_inputs["Age"] = st.sidebar.slider(
    "Age", 14.0, 61.0,
    value=float(st.session_state.get("Age", default_values["Age"])),
    step=1.0
)

feature_inputs["Height"] = st.sidebar.slider(
    "Height (m)", 1.45, 1.98,
    value=float(st.session_state.get("Height", default_values["Height"])), 
    step=0.01
)

feature_inputs["Weight"] = st.sidebar.slider(
    "Weight (kg)", 39.0, 173.0,
    value=float(st.session_state.get("Weight", default_values["Weight"])),
    step=1.0
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
    "Water consumption (1-3 scale)", 1.0, 3.0,
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

# Fungsi preprocessing yang lebih sederhana dan robust
def simple_preprocessing(feature_dict):
    """Preprocessing sederhana yang sesuai dengan model"""
    
    # Mapping categorical to numerical
    categorical_mapping = {
        'Gender': {'Female': 0, 'Male': 1},
        'FHWO': {'no': 0, 'yes': 1},
        'FAVC': {'no': 0, 'yes': 1},
        'CAEC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'SMOKE': {'no': 0, 'yes': 1},
        'SCC': {'no': 0, 'yes': 1},
        'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3},
        'MTRANS': {
            'Public_Transportation': 0,
            'Walking': 1, 
            'Automobile': 2,
            'Motorbike': 3,
            'Bike': 4
        }
    }
    
    # Process features
    processed = {}
    
    # Numerical features langsung
    numerical_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for feature in numerical_features:
        processed[feature] = feature_dict[feature]
    
    # Categorical features mapping
    for feature, mapping in categorical_mapping.items():
        processed[feature] = mapping[feature_dict[feature]]
    
    # Create DataFrame dengan urutan yang konsisten
    feature_order = ['Gender', 'Age', 'Height', 'Weight', 'FHWO', 'FAVC', 'FCVC', 
                    'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS']
    
    input_array = np.array([[processed[feature] for feature in feature_order]])
    
    return input_array, feature_order

# Main content area
tab1, tab2 = st.tabs(["📊 Prediction", "ℹ️ About"])

with tab1:
    st.header("📈 Prediction Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Features")
        display_df = pd.DataFrame([feature_inputs])
        st.dataframe(display_df, use_container_width=True)
        
        # BMI Analysis
        st.metric("Body Mass Index (BMI)", f"{bmi:.1f}")
        if bmi < 18.5:
            st.info("📊 BMI Category: Underweight")
        elif bmi < 25:
            st.success("📊 BMI Category: Normal weight")
        elif bmi < 30:
            st.warning("📊 BMI Category: Overweight")
        else:
            st.error("📊 BMI Category: Obesity")
    
    with col2:
        st.subheader("Preprocessed Features")
        input_array, feature_order = simple_preprocessing(feature_inputs)
        preprocessed_df = pd.DataFrame(input_array, columns=feature_order)
        st.dataframe(preprocessed_df, use_container_width=True)
    
    # Prediction button
    if st.button("🎯 Predict Obesity Level", type="primary", use_container_width=True):
        with st.spinner("Analyzing features and making prediction..."):
            try:
                # Convert to DMatrix untuk XGBoost
                dmatrix = xgb.DMatrix(input_array)
                
                # Make prediction
                prediction = model.predict(dmatrix)
                
                # Untuk multiclass, ambil class dengan probability tertinggi
                if len(prediction.shape) > 1:
                    pred_proba = prediction[0]
                    pred_class = np.argmax(pred_proba)
                else:
                    pred_class = int(prediction[0])
                    pred_proba = [1.0 if i == pred_class else 0.0 for i in range(7)]
                
                class_mapping = {
                    0: "Insufficient_Weight",
                    1: "Normal_Weight", 
                    2: "Overweight_Level_I",
                    3: "Overweight_Level_II", 
                    4: "Obesity_Type_I",
                    5: "Obesity_Type_II",
                    6: "Obesity_Type_III"
                }
                
                predicted_label = class_mapping.get(pred_class, f"Class {pred_class}")
                
                st.subheader("🎯 Prediction Result")
                
                # Determine CSS class based on prediction
                if "Insufficient_Weight" in predicted_label:
                    prediction_class = "insufficient-weight"
                    prediction_icon = "💪"
                elif "Normal_Weight" in predicted_label:
                    prediction_class = "normal-weight" 
                    prediction_icon = "✅"
                elif "Overweight" in predicted_label:
                    prediction_class = "overweight"
                    prediction_icon = "📊"
                else:
                    prediction_class = "obesity"
                    prediction_icon = "⚠️"
                
                st.markdown(f"""
                <div class="prediction-box {prediction_class}">
                    <h2 style="text-align: center; margin: 0;">
                        {prediction_icon} {predicted_label.replace('_', ' ')} {prediction_icon}
                    </h2>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("📊 Prediction Probabilities")
                
                # Create probability chart
                obesity_levels = [class_mapping[i].replace('_', ' ') for i in range(7)]
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Colors based on obesity level
                colors = ['#4ECDC4', '#45B7D1', '#FFD166', '#FF9F1C', '#FF6B6B', '#EE4266', '#C44569']
                bars = ax.bar(obesity_levels, pred_proba, color=colors, alpha=0.8)
                
                # Highlight predicted class
                bars[pred_class].set_edgecolor('black')
                bars[pred_class].set_linewidth(3)
                bars[pred_class].set_alpha(1.0)
                
                ax.set_ylabel('Probability')
                ax.set_title('Obesity Level Probabilities')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 0.01:  # Only show label if probability > 1%
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Probability table
                prob_df = pd.DataFrame({
                    'Obesity Level': obesity_levels,
                    'Probability': [f"{p:.4f}" for p in pred_proba]
                })
                
                # Highlight the predicted row
                def highlight_row(row):
                    if row.name == pred_class:
                        return ['background-color: #ffffcc'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(prob_df.style.apply(highlight_row, axis=1), use_container_width=True)
                
                # Health recommendations
                st.subheader("💡 Health Recommendations")
                
                recommendations = {
                    "Insufficient_Weight": [
                        "Increase caloric intake with nutrient-dense foods",
                        "Focus on strength training exercises", 
                        "Consult with a nutritionist for weight gain plan",
                        "Eat frequent, smaller meals throughout the day"
                    ],
                    "Normal_Weight": [
                        "Maintain balanced diet and regular exercise",
                        "Continue healthy lifestyle habits",
                        "Monitor weight regularly", 
                        "Stay physically active"
                    ],
                    "Overweight_Level_I": [
                        "Moderate calorie reduction",
                        "Increase physical activity to 150+ minutes per week",
                        "Focus on whole foods and reduce processed foods",
                        "Consider consulting a dietitian"
                    ],
                    "Overweight_Level_II": [
                        "Significant lifestyle changes needed",
                        "Aim for 300+ minutes of exercise weekly", 
                        "Strict dietary monitoring",
                        "Medical consultation recommended"
                    ],
                    "Obesity_Type_I": [
                        "Comprehensive weight management program",
                        "Medical supervision recommended",
                        "Structured exercise program",
                        "Behavioral therapy consideration"
                    ],
                    "Obesity_Type_II": [
                        "Immediate medical intervention",
                        "Structured weight loss program under supervision", 
                        "Possible pharmacological treatment",
                        "Regular health monitoring"
                    ],
                    "Obesity_Type_III": [
                        "Urgent medical attention required",
                        "Comprehensive treatment plan",
                        "Possible surgical interventions considered",
                        "Multidisciplinary team approach needed"
                    ]
                }
                
                for rec in recommendations.get(class_mapping[pred_class], []):
                    st.write(f"• {rec}")
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("💡 Tips: Make sure all input values are within the specified ranges.")

with tab2:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🏥 Obesity Risk Prediction System
    
    This application predicts obesity levels based on lifestyle and physical characteristics 
    using a trained XGBoost machine learning model.
    
    #### 📋 Features Used:
    - **Personal Information**: Gender, Age, Height, Weight
    - **Dietary Habits**: Vegetable consumption, meal frequency, high-calorie food intake
    - **Lifestyle Factors**: Physical activity, water intake, technology usage
    - **Health Indicators**: Family history, smoking, alcohol consumption
    
    #### 🎯 Obesity Levels:
    - Insufficient Weight
    - Normal Weight  
    - Overweight Level I & II
    - Obesity Type I, II & III
    
    #### 🔧 Technical Details:
    - **Model**: XGBoost Classifier
    - **Training Data**: Obesity dataset with multiple lifestyle factors
    - **Accuracy**: High predictive performance for obesity classification
    """)

# Test examples
st.sidebar.header("🧪 Test Examples")

example_inputs = {
    "Select an example...": default_values,
    "🏃 Healthy Lifestyle": {
        "Gender": "Male", "Age": 28.0, "Height": 1.78, "Weight": 72.0,
        "FCVC": 2.8, "NCP": 3.0, "CAEC": "Sometimes", "FAVC": "no",
        "CH2O": 2.5, "CALC": "Sometimes", "SCC": "yes", "FAF": 2.5,
        "TUE": 1.0, "MTRANS": "Walking", "FHWO": "no", "SMOKE": "no"
    },
    "⚠️ Obesity Risk": {
        "Gender": "Female", "Age": 42.0, "Height": 1.62, "Weight": 88.0,
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

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🏥 Obesity Prediction System | Made with Streamlit & XGBoost"
    "</div>",
    unsafe_allow_html=True
)
