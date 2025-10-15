import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import requests
import tempfile
import os
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from matplotlib.patches import Circle, Wedge, Rectangle

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
    .gauge-container {
        background: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .risk-legend {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #ddd;
    }
    .legend-item {
        display: flex;
        align-items: center;
        margin: 5px 0;
    }
    .legend-color {
        width: 20px;
        height: 20px;
        margin-right: 10px;
        border-radius: 3px;
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

# Fungsi untuk membuat gauge chart
def create_gauge_chart(pred_class, pred_proba, class_mapping):
    """Membuat gauge chart yang menarik"""
    
    obesity_levels = list(class_mapping.values())
    current_level = obesity_levels[pred_class]
    confidence = pred_proba[pred_class] * 100
    
    # Warna untuk setiap level
    color_map = {
        'Insufficient_Weight': '#4ECDC4',
        'Normal_Weight': '#45B7D1', 
        'Overweight_Level_I': '#FFD166',
        'Overweight_Level_II': '#FF9F1C',
        'Obesity_Type_I': '#FF6B6B',
        'Obesity_Type_II': '#EE4266',
        'Obesity_Type_III': '#C44569'
    }
    
    fig = go.Figure()
    
    # Gauge chart
    fig.add_trace(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence: {current_level.replace('_', ' ')}", 'font': {'size': 20}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color_map[current_level]},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 20], 'color': '#e3f2fd'},
                {'range': [20, 40], 'color': '#bbdefb'},
                {'range': [40, 60], 'color': '#90caf9'},
                {'range': [60, 80], 'color': '#64b5f6'},
                {'range': [80, 100], 'color': '#42a5f5'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# Fungsi untuk membuat radar chart
def create_radar_chart(feature_inputs):
    """Membuat radar chart untuk lifestyle factors"""
    
    categories = ['Physical Activity', 'Healthy Diet', 'Water Intake', 'Meal Frequency', 'Lifestyle Score']
    
    # Normalize values untuk radar chart
    physical_activity = (feature_inputs['FAF'] / 3.0) * 100
    healthy_diet = ((feature_inputs['FCVC'] - 1) / 2.0) * 100  # FCVC 1-3 -> 0-100
    water_intake = ((feature_inputs['CH2O'] - 1) / 2.0) * 100  # CH2O 1-3 -> 0-100
    meal_frequency = ((feature_inputs['NCP'] - 1) / 3.0) * 100  # NCP 1-4 -> 0-100
    
    # Lifestyle score (composite)
    lifestyle_score = (physical_activity + healthy_diet + water_intake + meal_frequency) / 4
    
    values = [physical_activity, healthy_diet, water_intake, meal_frequency, lifestyle_score]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # Close the radar
        theta=categories + [categories[0]],
        fill='toself',
        name='Lifestyle Factors',
        line=dict(color='#1f77b4'),
        fillcolor='rgba(31, 119, 180, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        height=400,
        title="Lifestyle Factors Radar Chart"
    )
    
    return fig

# Fungsi untuk membuat donut chart probabilities
def create_donut_chart(pred_proba, class_mapping):
    """Membuat donut chart untuk probabilities"""
    
    obesity_levels = [class_mapping[i].replace('_', ' ') for i in range(len(pred_proba))]
    colors = ['#4ECDC4', '#45B7D1', '#FFD166', '#FF9F1C', '#FF6B6B', '#EE4266', '#C44569']
    
    fig = go.Figure(data=[go.Pie(
        labels=obesity_levels,
        values=pred_proba,
        hole=.5,
        marker_colors=colors,
        textinfo='label+percent',
        insidetextorientation='radial'
    )])
    
    fig.update_layout(
        title="Obesity Level Probabilities",
        height=400,
        showlegend=False
    )
    
    return fig

# Fungsi untuk membuat risk meter dengan legend
def create_risk_meter_with_legend(pred_class):
    """Membuat risk meter visual dengan legend terpisah"""
    
    risk_info = [
        {'level': 0, 'label': 'Very Low', 'color': '#4ECDC4', 'description': 'Underweight'},
        {'level': 1, 'label': 'Low', 'color': '#45B7D1', 'description': 'Normal Weight'},
        {'level': 2, 'label': 'Moderate', 'color': '#FFD166', 'description': 'Overweight I'},
        {'level': 3, 'label': 'High', 'color': '#FF9F1C', 'description': 'Overweight II'},
        {'level': 4, 'label': 'Very High', 'color': '#FF6B6B', 'description': 'Obesity I'},
        {'level': 5, 'label': 'Severe', 'color': '#EE4266', 'description': 'Obesity II'},
        {'level': 6, 'label': 'Critical', 'color': '#C44569', 'description': 'Obesity III'}
    ]
    
    current_risk = risk_info[pred_class]
    
    # Create risk meter
    fig = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        value = pred_class,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': 1},
        number = {
            'font': {'size': 24, 'color': current_risk['color']},
            'prefix': 'Level ',
            'suffix': f" - {current_risk['label']}"
        },
        gauge = {
            'shape': "bullet",
            'axis': {'range': [0, 6], 'tickwidth': 1, 'tickvals': list(range(7))},
            'threshold': {
                'line': {'color': "black", 'width': 3},
                'thickness': 0.8,
                'value': pred_class},
            'steps': [
                {'range': [0, 1], 'color': '#4ECDC4'},
                {'range': [1, 2], 'color': '#45B7D1'},
                {'range': [2, 3], 'color': '#FFD166'},
                {'range': [3, 4], 'color': '#FF9F1C'},
                {'range': [4, 5], 'color': '#FF6B6B'},
                {'range': [5, 6], 'color': '#EE4266'},
                {'range': [6, 7], 'color': '#C44569'}],
            'bar': {'color': "black", 'thickness': 0.8}}))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=50, b=10),
        title=f"Obesity Risk Level: {current_risk['description']}"
    )
    
    return fig, risk_info

# Fungsi untuk membuat legend
def create_risk_legend(risk_info):
    """Membuat legend untuk risk meter"""
    
    legend_html = """
    <div class="risk-legend">
        <h4>🟰 Risk Level Legend</h4>
    """
    
    for risk in risk_info:
        legend_html += f"""
        <div class="legend-item">
            <div class="legend-color" style="background-color: {risk['color']};"></div>
            <div>
                <strong>Level {risk['level']}: {risk['label']}</strong><br>
                <small>{risk['description']}</small>
            </div>
        </div>
        """
    
    legend_html += "</div>"
    return legend_html

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

# Fungsi preprocessing yang sesuai dengan expected features model
def correct_preprocessing(feature_dict):
    """Preprocessing yang sesuai dengan expected features model"""
    
    # Mapping categorical to numerical untuk remainder features
    categorical_to_numerical = {
        "FHWO": {"no": 0, "yes": 1},
        "FAVC": {"no": 0, "yes": 1},
        "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
        "SMOKE": {"no": 0, "yes": 1},
        "SCC": {"no": 0, "yes": 1},
        "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    }
    
    # Create DataFrame dengan expected features
    expected_features = [
        'ohe__Gender_Male', 'ohe__MTRANS_Bike', 'ohe__MTRANS_Motorbike', 
        'ohe__MTRANS_Public_Transportation', 'ohe__MTRANS_Walking', 
        'remainder__Age', 'remainder__Height', 'remainder__Weight', 
        'remainder__FHWO', 'remainder__FAVC', 'remainder__FCVC', 
        'remainder__NCP', 'remainder__CAEC', 'remainder__SMOKE', 
        'remainder__CH2O', 'remainder__SCC', 'remainder__FAF', 
        'remainder__TUE', 'remainder__CALC'
    ]
    
    # Initialize semua features dengan 0
    processed_data = {feature: 0.0 for feature in expected_features}
    
    # Set one-hot encoding untuk Gender
    if feature_dict["Gender"] == "Male":
        processed_data["ohe__Gender_Male"] = 1.0
    else:
        processed_data["ohe__Gender_Male"] = 0.0
    
    # Set one-hot encoding untuk MTRANS
    mtrans_mapping = {
        "Bike": "ohe__MTRANS_Bike",
        "Motorbike": "ohe__MTRANS_Motorbike", 
        "Public_Transportation": "ohe__MTRANS_Public_Transportation",
        "Walking": "ohe__MTRANS_Walking",
        "Automobile": "ohe__MTRANS_Automobile"
    }
    
    # Reset semua MTRANS features ke 0
    for feature in ['ohe__MTRANS_Bike', 'ohe__MTRANS_Motorbike', 
                   'ohe__MTRANS_Public_Transportation', 'ohe__MTRANS_Walking']:
        processed_data[feature] = 0.0
    
    # Set yang aktif berdasarkan pilihan
    if feature_dict["MTRANS"] in mtrans_mapping:
        feature_name = mtrans_mapping[feature_dict["MTRANS"]]
        if feature_name in processed_data:
            processed_data[feature_name] = 1.0
    
    # Set numerical features
    processed_data["remainder__Age"] = float(feature_dict["Age"])
    processed_data["remainder__Height"] = float(feature_dict["Height"])
    processed_data["remainder__Weight"] = float(feature_dict["Weight"])
    processed_data["remainder__FCVC"] = float(feature_dict["FCVC"])
    processed_data["remainder__NCP"] = float(feature_dict["NCP"])
    processed_data["remainder__CH2O"] = float(feature_dict["CH2O"])
    processed_data["remainder__FAF"] = float(feature_dict["FAF"])
    processed_data["remainder__TUE"] = float(feature_dict["TUE"])
    
    # Set encoded categorical features
    processed_data["remainder__FHWO"] = categorical_to_numerical["FHWO"][feature_dict["FHWO"]]
    processed_data["remainder__FAVC"] = categorical_to_numerical["FAVC"][feature_dict["FAVC"]]
    processed_data["remainder__CAEC"] = categorical_to_numerical["CAEC"][feature_dict["CAEC"]]
    processed_data["remainder__SMOKE"] = categorical_to_numerical["SMOKE"][feature_dict["SMOKE"]]
    processed_data["remainder__SCC"] = categorical_to_numerical["SCC"][feature_dict["SCC"]]
    processed_data["remainder__CALC"] = categorical_to_numerical["CALC"][feature_dict["CALC"]]
    
    # Create DataFrame dengan urutan yang tepat
    input_df = pd.DataFrame([processed_data])[expected_features]
    
    return input_df

# Main content area
tab1, tab2 = st.tabs(["🎯 Prediction Dashboard", "ℹ️ About"])

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
        input_df = correct_preprocessing(feature_inputs)
        st.dataframe(input_df.T.rename(columns={0: "Value"}), use_container_width=True)
    
    # Prediction button
    if st.button("🎯 Predict Obesity Level", type="primary", use_container_width=True):
        with st.spinner("Analyzing features and making prediction..."):
            try:
                # Convert to DMatrix untuk XGBoost
                dmatrix = xgb.DMatrix(input_df)
                
                # Make prediction
                prediction = model.predict(dmatrix)
                
                # Untuk multiclass, ambil class dengan probability tertinggi
                if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                    pred_proba = prediction[0]
                    pred_class = np.argmax(pred_proba)
                else:
                    # Jika binary classification atau regression
                    pred_class = int(prediction[0])
                    # Buat probability array dummy
                    pred_proba = np.zeros(7)
                    pred_proba[pred_class] = 1.0
                
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
                
                # VISUALISASI BARU - Dashboard dengan berbagai chart
                st.subheader("📊 Advanced Visualization Dashboard")
                
                # Row 1: Gauge Chart dan Risk Meter dengan Legend
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_gauge_chart(pred_class, pred_proba, class_mapping), 
                                  use_container_width=True)
                
                with col2:
                    # Buat risk meter dan dapatkan info risk
                    risk_fig, risk_info = create_risk_meter_with_legend(pred_class)
                    st.plotly_chart(risk_fig, use_container_width=True)
                    
                    # Tampilkan legend di bawah risk meter
                    st.markdown(create_risk_legend(risk_info), unsafe_allow_html=True)
                
                # Row 2: Donut Chart dan Radar Chart
                col3, col4 = st.columns(2)
                
                with col3:
                    st.plotly_chart(create_donut_chart(pred_proba, class_mapping), 
                                  use_container_width=True)
                
                with col4:
                    st.plotly_chart(create_radar_chart(feature_inputs), 
                                  use_container_width=True)
                
                # Row 3: Traditional Bar Chart (sebagai backup)
                st.subheader("📈 Probability Distribution")
                obesity_levels = [class_mapping[i].replace('_', ' ') for i in range(len(pred_proba))]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                colors = ['#4ECDC4', '#45B7D1', '#FFD166', '#FF9F1C', '#FF6B6B', '#EE4266', '#C44569']
                bars = ax.bar(obesity_levels, pred_proba, color=colors, alpha=0.8)
                
                # Highlight predicted class
                if pred_class < len(bars):
                    bars[pred_class].set_edgecolor('black')
                    bars[pred_class].set_linewidth(3)
                    bars[pred_class].set_alpha(1.0)
                
                ax.set_ylabel('Probability')
                ax.set_title('Obesity Level Probabilities')
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 0.01:
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
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
                
                rec_key = class_mapping[pred_class]
                for rec in recommendations.get(rec_key, []):
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
