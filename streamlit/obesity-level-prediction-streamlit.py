import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Konfigurasi halaman
st.set_page_config(
    page_title="Obesity Level Prediction",
    page_icon="🏥",
    layout="wide"
)

# Load model dari URL
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/streamlit/xgb-obesity.json"
    try:
        response = requests.get(url)
        model_json = response.json()
        model = xgb.XGBClassifier()
        model.load_model(model_json)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocessing function
def preprocess_input(data):
    # Copy data untuk menghindari warning
    df = data.copy()
    
    # Encoding untuk categorical variables
    label_encoders = {}
    
    categorical_columns = ['Gender', 'FHWO', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
    
    # Mapping untuk categorical features
    encoding_maps = {
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
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].map(encoding_maps[col])
    
    return df

# Fungsi untuk membuat visualisasi
def create_visualizations(input_data, prediction, prediction_proba):
    # Obesity level labels
    obesity_levels = [
        'Insufficient_Weight',
        'Normal_Weight',
        'Overweight_Level_I',
        'Overweight_Level_II',
        'Obesity_Type_I',
        'Obesity_Type_II',
        'Obesity_Type_III'
    ]
    
    # 1. Grafik Probabilitas Prediksi
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF']
    bars = ax1.bar(obesity_levels, prediction_proba[0], color=colors)
    
    # Highlight predicted class
    predicted_idx = np.argmax(prediction_proba[0])
    bars[predicted_idx].set_edgecolor('black')
    bars[predicted_idx].set_linewidth(3)
    
    ax1.set_title('Probability Distribution for Obesity Levels', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Obesity Level', fontsize=12)
    ax1.set_ylabel('Probability', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Tambah nilai di atas bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 2. Grafik Lifestyle Factors
    lifestyle_data = {
        'Factors': ['High Caloric Food', 'Vegetable Intake', 'Between Meals', 
                   'Smoking', 'Water Intake', 'Calorie Monitoring', 'Physical Activity'],
        'Values': [
            input_data['FAVC'].iloc[0],
            input_data['FCVC'].iloc[0],
            1 if input_data['CAEC'].iloc[0] > 0 else 0,
            input_data['SMOKE'].iloc[0],
            input_data['CH2O'].iloc[0],
            input_data['SCC'].iloc[0],
            input_data['FAF'].iloc[0]
        ]
    }
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(lifestyle_data['Factors']))
    ax2.barh(y_pos, lifestyle_data['Values'], color='skyblue')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(lifestyle_data['Factors'])
    ax2.set_xlabel('Value')
    ax2.set_title('Lifestyle Factors Analysis', fontsize=16, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # 3. Pie Chart untuk Categorical Features
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gender
    gender_labels = ['Female', 'Male']
    gender_values = [1 - input_data['Gender'].iloc[0], input_data['Gender'].iloc[0]]
    axes[0,0].pie(gender_values, labels=gender_labels, autopct='%1.1f%%', colors=['lightpink', 'lightblue'])
    axes[0,0].set_title('Gender Distribution')
    
    # Family History
    fhwo_labels = ['No', 'Yes']
    fhwo_values = [1 - input_data['FHWO'].iloc[0], input_data['FHWO'].iloc[0]]
    axes[0,1].pie(fhwo_values, labels=fhwo_labels, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
    axes[0,1].set_title('Family History of Overweight')
    
    # Transportation
    trans_mapping = {0: 'Public Transport', 1: 'Walking', 2: 'Automobile', 3: 'Motorbike', 4: 'Bike'}
    trans_label = trans_mapping.get(input_data['MTRANS'].iloc[0], 'Unknown')
    axes[1,0].pie([1], labels=[trans_label], autopct='%1.1f%%', colors=['gold'])
    axes[1,0].set_title('Primary Transportation')
    
    # Alcohol Consumption
    calc_mapping = {0: 'Never', 1: 'Sometimes', 2: 'Frequently', 3: 'Always'}
    calc_label = calc_mapping.get(input_data['CALC'].iloc[0], 'Unknown')
    axes[1,1].pie([1], labels=[calc_label], autopct='%1.1f%%', colors=['orange'])
    axes[1,1].set_title('Alcohol Consumption')
    
    plt.tight_layout()
    
    return fig1, fig2, fig3

# Main aplikasi
def main():
    st.title("🏥 Obesity Level Prediction System")
    st.markdown("""
    This application predicts obesity levels based on lifestyle and physical characteristics 
    using a trained XGBoost machine learning model.
    """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load the model. Please check the model URL.")
        return
    
    # Sidebar untuk input
    st.sidebar.header("📊 Input Parameters")
    
    # Personal Information
    st.sidebar.subheader("Personal Information")
    gender = st.sidebar.selectbox("Gender", ['Female', 'Male'])
    age = st.sidebar.slider("Age", min_value=14.0, max_value=61.0, value=25.0, step=1.0)
    height = st.sidebar.slider("Height (m)", min_value=1.45, max_value=1.98, value=1.70, step=0.01)
    weight = st.sidebar.slider("Weight (kg)", min_value=39.0, max_value=173.0, value=70.0, step=1.0)
    
    # Family History
    st.sidebar.subheader("Family History")
    fhwo = st.sidebar.selectbox("Has a family member suffered or suffers from overweight?", ['yes', 'no'])
    
    # Dietary Habits
    st.sidebar.subheader("Dietary Habits")
    favc = st.sidebar.selectbox("Do you eat high caloric food frequently?", ['no', 'yes'])
    fcvc = st.sidebar.slider("Do you usually eat vegetables in your meals? (1-3)", 
                            min_value=1.0, max_value=3.0, value=2.0, step=1.0)
    ncp = st.sidebar.slider("How many main meals do you have daily?", 
                           min_value=1.0, max_value=4.0, value=3.0, step=1.0)
    caec = st.sidebar.selectbox("Do you eat any food between meals?", 
                               ['no', 'Sometimes', 'Frequently', 'Always'])
    
    # Lifestyle Habits
    st.sidebar.subheader("Lifestyle Habits")
    smoke = st.sidebar.selectbox("Do you smoke?", ['no', 'yes'])
    ch2o = st.sidebar.slider("How much water do you drink daily? (1-3)", 
                            min_value=1.0, max_value=3.0, value=2.0, step=1.0)
    scc = st.sidebar.selectbox("Do you monitor the calories you eat daily?", ['no', 'yes'])
    faf = st.sidebar.slider("How often do you have physical activity? (0-3)", 
                           min_value=0.0, max_value=3.0, value=1.0, step=1.0)
    tue = st.sidebar.slider("How much time do you use technological devices? (0-2)", 
                           min_value=0.0, max_value=2.0, value=1.0, step=1.0)
    calc = st.sidebar.selectbox("How often do you drink alcohol?", 
                               ['no', 'Sometimes', 'Frequently', 'Always'])
    mtrans = st.sidebar.selectbox("Which transportation do you usually use?", 
                                 ['Public_Transportation', 'Walking', 'Automobile', 'Motorbike', 'Bike'])
    
    # Calculate BMI
    bmi = weight / (height ** 2)
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'FHWO': [fhwo],
        'FAVC': [favc],
        'FCVC': [fcvc],
        'NCP': [ncp],
        'CAEC': [caec],
        'SMOKE': [smoke],
        'CH2O': [ch2o],
        'SCC': [scc],
        'FAF': [faf],
        'TUE': [tue],
        'CALC': [calc],
        'MTRANS': [mtrans]
    })
    
    # Tombol prediksi
    if st.sidebar.button("🚀 Predict Obesity Level", use_container_width=True):
        with st.spinner("Predicting..."):
            # Preprocess input
            processed_data = preprocess_input(input_data)
            
            # Make prediction
            prediction = model.predict(processed_data)
            prediction_proba = model.predict_proba(processed_data)
            
            # Obesity level mapping
            obesity_levels = [
                'Insufficient_Weight',
                'Normal_Weight',
                'Overweight_Level_I',
                'Overweight_Level_II',
                'Obesity_Type_I',
                'Obesity_Type_II',
                'Obesity_Type_III'
            ]
            
            predicted_level = obesity_levels[prediction[0]]
            
            # Display results
            st.success("✅ Prediction Completed!")
            
            # Main results section
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Obesity Level", predicted_level)
            
            with col2:
                st.metric("BMI", f"{bmi:.2f}")
                
                # BMI Interpretation
                if bmi < 18.5:
                    bmi_category = "Underweight"
                elif bmi < 25:
                    bmi_category = "Normal weight"
                elif bmi < 30:
                    bmi_category = "Overweight"
                else:
                    bmi_category = "Obesity"
                st.metric("BMI Category", bmi_category)
            
            with col3:
                confidence = np.max(prediction_proba[0]) * 100
                st.metric("Prediction Confidence", f"{confidence:.1f}%")
            
            # Create visualizations
            st.header("📈 Prediction Analysis")
            fig1, fig2, fig3 = create_visualizations(processed_data, prediction, prediction_proba)
            
            # Display charts
            st.subheader("Probability Distribution")
            st.pyplot(fig1)
            
            col4, col5 = st.columns(2)
            
            with col4:
                st.subheader("Lifestyle Factors")
                st.pyplot(fig2)
            
            with col5:
                st.subheader("Categorical Features")
                st.pyplot(fig3)
            
            # Detailed probabilities
            st.subheader("Detailed Probabilities")
            prob_df = pd.DataFrame({
                'Obesity Level': obesity_levels,
                'Probability': prediction_proba[0]
            }).sort_values('Probability', ascending=False)
            
            st.dataframe(prob_df.style.format({'Probability': '{:.4f}'}).highlight_max(axis=0, color='lightgreen'))
            
            # Health recommendations based on prediction
            st.subheader("💡 Health Recommendations")
            recommendations = {
                'Insufficient_Weight': [
                    "Increase caloric intake with nutrient-dense foods",
                    "Focus on strength training exercises",
                    "Consult with a nutritionist for weight gain plan",
                    "Eat frequent, smaller meals throughout the day"
                ],
                'Normal_Weight': [
                    "Maintain balanced diet and regular exercise",
                    "Continue healthy lifestyle habits",
                    "Monitor weight regularly",
                    "Stay physically active"
                ],
                'Overweight_Level_I': [
                    "Moderate calorie reduction",
                    "Increase physical activity to 150+ minutes per week",
                    "Focus on whole foods and reduce processed foods",
                    "Consider consulting a dietitian"
                ],
                'Overweight_Level_II': [
                    "Significant lifestyle changes needed",
                    "Aim for 300+ minutes of exercise weekly",
                    "Strict dietary monitoring",
                    "Medical consultation recommended"
                ],
                'Obesity_Type_I': [
                    "Comprehensive weight management program",
                    "Medical supervision recommended",
                    "Structured exercise program",
                    "Behavioral therapy consideration"
                ],
                'Obesity_Type_II': [
                    "Immediate medical intervention",
                    "Structured weight loss program under supervision",
                    "Possible pharmacological treatment",
                    "Regular health monitoring"
                ],
                'Obesity_Type_III': [
                    "Urgent medical attention required",
                    "Comprehensive treatment plan",
                    "Possible surgical interventions considered",
                    "Multidisciplinary team approach needed"
                ]
            }
            
            for rec in recommendations.get(predicted_level, []):
                st.write(f"• {rec}")

if __name__ == "__main__":
    main()
