'''
    This application program is dedicated to study purposes during 
    the Data Analytics/Data Science bootcamp batch 34 at Dibimbing.id. 
    This program was created as a final project as one of the graduation requirements.

    The program's objective: to predict obesity levels (7 levels) using 
    a trained machine learning model that has been tested and is the best.
    With this program, learners are expected to see how the model 
    would be implemented in a web-based application, such as Streamlit.

    Input: all features provided in the data.
    Output: The model's predicted obesity levels.

    (c) daudrusyadnurdin@gmail.com, 2025-10
'''
# -------------------------------
# Library used
# -------------------------------
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
import seaborn as sns

# -------------------------------------------
# Set page config
# -------------------------------------------
st.cache_data.clear()

st.set_page_config(
    page_title="Obesity Levels Prediction App", 
    layout="wide",
    page_icon="🏥"
)

# -------------------------------------------
# Custom CSS untuk styling
# -------------------------------------------
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
    .risk-legend-item {
        background: white;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 5px solid;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .health-score-excellent { color: #28a745; font-weight: bold; }
    .health-score-good { color: #ffc107; font-weight: bold; }
    .health-score-fair { color: #fd7e14; font-weight: bold; }
    .health-score-poor { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# Load model with error handling 
# ------------------------------------------------
@st.cache_resource
def load_model():
    url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/models/xgb_obesity.json"
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

# ----------------------------
# Main functions defined
# ----------------------------
def calculate_health_score(feature_inputs):
    """Calculate a simple health score based on user input"""
    
    score = 0
    
    # Physical Activity (30 points max)
    activity_score = (feature_inputs['FAF'] / 3.0) * 30
    score += activity_score
    
    # Diet Quality (30 points max)
    diet_score = (feature_inputs['FCVC'] / 3.0) * 20
    if feature_inputs['FAVC'] == 'no':
        diet_score += 10
    else:
        diet_score += 5
    score += diet_score
    
    # Healthy Habits (20 points max)
    habits_score = 0
    if feature_inputs['SMOKE'] == 'no':
        habits_score += 10
    if feature_inputs['CALC'] in ['no', 'Sometimes']:
        habits_score += 10
    score += habits_score
    
    # Lifestyle Balance (20 points max)
    lifestyle_score = (1 - feature_inputs['TUE'] / 2.0) * 20
    score += lifestyle_score
    
    return max(0, min(100, score))

def create_gauge_chart(pred_class, pred_proba, class_mapping):
    """Create an engaging gauge chart"""
    
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

def safe_float_convert(value, default=0):
    """Safe conversion to float dengan handling yes/no"""
    try:
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ['yes', 'true', '1', 'always']:
                return 1.0
            elif value in ['no', 'false', '0', 'never']:
                return 0.0
            elif value in ['sometimes']:
                return 0.5
            elif value in ['frequently']:
                return 0.8
        return float(value)
    except (ValueError, TypeError):
        return float(default)

# -----------------
# Radam diagram
# -----------------
def create_health_radar(feature_inputs):
    """Radar chart khusus Health Indicators"""
    categories = [
        'Smoking (SMOKE)',
        'Alcohol (CALC)',  
        'Family History (FHWO)'
    ]
    
    values = [
        # SMOKE: Lower is better (binary, reversed)
        (1 - safe_float_convert(feature_inputs.get('SMOKE', 'no'))) * 5,
        
        # CALC: Lower is better (0-3 → 0-5, reversed)
        (1 - (safe_float_convert(feature_inputs.get('CALC', 'no')) / 3)) * 5,
        
        # FHWO: Lower is better (binary, reversed)  
        (1 - safe_float_convert(feature_inputs.get('FHWO', 'no'))) * 5,
    ]
    
    #ax.set_theta_offset(np.pi / 2)
    
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        line=dict(color='#dc3545'), fillcolor='rgba(220, 53, 69, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title_text="❤️ HEALTH INDICATORS",
        title_font_size=20,
        title_x=0.0, 
        height=400
    )
    return fig

# Dietary Habits: Vegetable consumption, Meal frequency, High-calorie food intake, Food between meals, Calorie monitoring
def create_dietary_radar(feature_inputs):
    """Radar chart khusus Dietary Habits"""
    categories = [
        'Vegetable Consumption (FCVC)',
        'Meal Frequency (NCP)', 
        'High-Calorie Food (FAVC)',
        'Snacking (CAEC)',           
        'Calorie Monitoring (SCC)'   
    ]
    
    values = [
        # FCVC: Higher is better (1-3 → 0-5)
        ((safe_float_convert(feature_inputs.get('FCVC', 2)) - 1) / 2) * 5,
        
        # NCP: Optimal around middle (1-4 → 0-5)
        ((safe_float_convert(feature_inputs.get('NCP', 2.5)) - 1) / 3) * 5,
        
        # FAVC: Lower is better (binary, reversed)
        (1 - safe_float_convert(feature_inputs.get('FAVC', 'no'))) * 5,
        
        # CAEC: Lower is better (0-3 → 0-5, reversed)
        (1 - (safe_float_convert(feature_inputs.get('CAEC', 'no')) / 3)) * 5,
        
        # SCC: Lower is better (binary, reversed)
        (1 - safe_float_convert(feature_inputs.get('SCC', 'no'))) * 5  
    ]
    
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill='toself',
        line=dict(color='#28a745'), fillcolor='rgba(40, 167, 69, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title_text="🍽️ DIETARY HABITS",
        title_font_size=20,
        title_x=0.0,
        height=400
    )
    return fig
    
# Lifestyle Factors: Physical activity, Water intake, Technology usage, Transportation
def create_lifestyle_radar(feature_inputs):
    """Radar chart khusus Lifestyle Factors"""
    categories = [
        'Physical Activity (FAF)',
        'Water Intake (CH2O)',
        'Technology Usage (TUE)',
        'Transportation (MTRANS)'
    ]
    
    values = [
        # FAF: Higher is better (0-3 → 0-5)
        (safe_float_convert(feature_inputs.get('FAF', 1.5)) / 3) * 5,
        
        # CH2O: Higher is better (1-3 → 0-5)
        ((safe_float_convert(feature_inputs.get('CH2O', 2)) - 1) / 2) * 5,
        
        # TUE: Lower is better (0-2 → 0-5, reversed)
        (1 - (safe_float_convert(feature_inputs.get('TUE', 1)) / 2)) * 5
    ]
    
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=categories, fill='toself', 
        line=dict(color='#17a2b8'), fillcolor='rgba(23, 162, 184, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        title_text="🏃‍♂️ LIFESTYLE FACTORS", 
        title_font_size=20,
        title_x=0.0,
        height=400
    )
    return fig

# -----------------
# Donut diagram
# -----------------
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
        #title="Obesity Level Probabilities",
        height=400,
        showlegend=False
    )
    
    return fig

def create_risk_meter_with_legend(pred_class):
    """Membuat risk meter visual dengan legend"""
    
    risk_info = [
        {'level': 0, 'label': 'Very Low', 'color': '#4ECDC4', 'description': 'Underweight'},
        {'level': 1, 'label': 'Low', 'color': '#45B7D1', 'description': 'Normal Weight'},
        {'level': 2, 'label': 'Moderate', 'color': '#FFD166', 'description': 'Overweight Level I'},
        {'level': 3, 'label': 'High', 'color': '#FF9F1C', 'description': 'Overweight Level II'},
        {'level': 4, 'label': 'Very High', 'color': '#FF6B6B', 'description': 'Obesity Type I'},
        {'level': 5, 'label': 'Severe', 'color': '#EE4266', 'description': 'Obesity Type II'},
        {'level': 6, 'label': 'Critical', 'color': '#C44569', 'description': 'Obesity Type III'}
    ]
    
    current_risk = risk_info[pred_class]
    
    fig = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        value = pred_class,
        domain = {'x': [0, 1], 'y': [0, 1]},
        delta = {'reference': 1},
        number = {
            'font': {'size': 18, 'color': current_risk['color']},
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
        height=250,
        margin=dict(l=10, r=10, t=60, b=10),
        title=f"Obesity Level: {current_risk['description']}"
    )
    
    return fig, risk_info

def display_color_bar_legend(risk_info):
    """Menampilkan color bar horizontal yang informatif"""
    
    st.markdown("**Obesity Classification**")
    
    # Create color gradient bar
    color_gradient = "background: linear-gradient(90deg"
    for risk in risk_info:
        color_gradient += f", {risk['color']}"
    color_gradient += ");"
    
    # Mapping yang jelas
    class_mapping = {
        'Very Low': 'Insufficient Weight',
        'Low': 'Normal Weight', 
        'Moderate': 'Overweight Level I',
        'High': 'Overweight Level II',
        'Very High': 'Obesity Type I',
        'Severe': 'Obesity Type II',
        'Critical': 'Obesity Type III'
    }
    
    st.markdown(
        f"""
        <div style='
            {color_gradient}
            height: 20px;
            border-radius: 4px;
            margin: 8px 0 2px 0;
            border: 1px solid #ddd;
        '></div>
        
        <!-- Risk Levels (Top Row) -->
        <div style='display: flex; justify-content: space-between; font-size: 10px; font-weight: bold; margin: 5px 0 2px 0;'>
            {"".join([f"<div style='color: {risk['color']}; text-align: center; width: {100/len(risk_info)}%'>{risk['label']}</div>" for risk in risk_info])}
        </div>
        
        <!-- Obesity Classes (Bottom Row) -->
        <div style='display: flex; justify-content: space-between; font-size: 10px; color: #666; margin-bottom: 10px;'>
            {"".join([f"<div style='text-align: center; width: {100/len(risk_info)}%'>{class_mapping[risk['label']]}</div>" for risk in risk_info])}
        </div>
        """,
        unsafe_allow_html=True
    )
    

# ============================
# MAIN PROGRAM
# ============================

# -------------------
# Banner: aksesoris
# -------------------
header_image_url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/assets/obelution-forbidden-bw.png"
st.image(header_image_url, use_container_width=True)

# -------------------
# Header aplikasi
# -------------------
st.markdown('<h1 class="main-header">🏥 Obesity Level Prediction App</h1>', unsafe_allow_html=True)

st.markdown(
    '<h3 style="text-align: center; color: #FFA07A; margin-bottom: 2rem;">'
    'Predict obesity levels based on your input parameters using XGBoost machine learning model'
    '</h3>', 
    unsafe_allow_html=True
)

# -------------------
# Load model
# -------------------
model = load_model()

if model is None:
    st.error("Failed to load model. Please check the model URL.")
    st.stop()

# -------------------
# Default values
# -------------------
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

# -------------------------
# Initialize session state
# -------------------------
for key in default_values.keys():
    if key not in st.session_state:
        st.session_state[key] = default_values[key]

# -----------------------------
# Sidebar untuk input features
# -----------------------------
st.sidebar.header("🛠️ Feature Configuration")

def reset_defaults():
    for k, v in default_values.items():
        st.session_state[k] = v

if st.sidebar.button("🔄 Reset to Default"):
    reset_defaults()
    st.rerun()

# --------------------
# Collect user inputs
# --------------------
feature_inputs = {} # Initial state

# Personal information
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


# Health Indicators: Family history with overweight, Smoking habit, alcohol consumption
fhwo_options = ["no", "yes"]
feature_inputs["FHWO"] = st.sidebar.selectbox(
    "Family history of overweight (FHWO)", 
    fhwo_options,
    index=fhwo_options.index(st.session_state.get("FHWO", default_values["FHWO"]))
)

smoke_options = ["no", "yes"]
feature_inputs["SMOKE"] = st.sidebar.selectbox(
    "Smoking habit (SMOKE)", 
    smoke_options,
    index=smoke_options.index(st.session_state.get("SMOKE", default_values["SMOKE"]))
)

calc_options = ["no", "Sometimes", "Frequently", "Always"]
feature_inputs["CALC"] = st.sidebar.selectbox(
    "Alcohol consumption (CALC)", 
    calc_options,
    index=calc_options.index(st.session_state.get("CALC", default_values["CALC"]))
)

# Dietary Habits: Vegetable consumption, Meal frequency, High-calorie food intake, 
# Food between meals, Calorie monitoring
feature_inputs["FCVC"] = st.sidebar.slider(
    "Frequency of vegetable consumption (FCVC)", 1.0, 3.0,
    value=float(st.session_state.get("FCVC", default_values["FCVC"])),
    step=0.1
)

feature_inputs["NCP"] = st.sidebar.slider(
    "Number of main meals per day (NCP)", 1.0, 4.0,
    value=float(st.session_state.get("NCP", default_values["NCP"])),
    step=0.1
)

favc_options = ["no", "yes"]
feature_inputs["FAVC"] = st.sidebar.selectbox(
    "Frequent consumption of high caloric food (FAVC)", 
    favc_options,
    index=favc_options.index(st.session_state.get("FAVC", default_values["FAVC"]))
)

caec_options = ["no", "Sometimes", "Frequently", "Always"]
feature_inputs["CAEC"] = st.sidebar.selectbox(
    "Consumption of food between meals (CAEC)", 
    caec_options,
    index=caec_options.index(st.session_state.get("CAEC", default_values["CAEC"]))
)

scc_options = ["no", "yes"]
feature_inputs["SCC"] = st.sidebar.selectbox(
    "Monitor calorie consumption (SCC)", 
    scc_options,
    index=scc_options.index(st.session_state.get("SCC", default_values["SCC"]))
)

# Lifestyle Factors: Physical activity, Water intake, Technology usage, Transportation
feature_inputs["FAF"] = st.sidebar.slider(
    "Physical activity frequency (FAF)", 0.0, 3.0,
    value=float(st.session_state.get("FAF", default_values["FAF"])),
    step=0.1
)

feature_inputs["CH2O"] = st.sidebar.slider(
    "Water consumption (CH2O)", 1.0, 3.0,
    value=float(st.session_state.get("CH2O", default_values["CH2O"])),
    step=0.1
)

feature_inputs["TUE"] = st.sidebar.slider(
    "Time using electronic devices (TUE)", 0.0, 2.0,
    value=float(st.session_state.get("TUE", default_values["TUE"])),
    step=0.1
)

mtrans_options = ["Public_Transportation", "Walking", "Automobile", "Motorbike", "Bike"]
feature_inputs["MTRANS"] = st.sidebar.selectbox(
    "Primary transportation method (MTRANS)", 
    mtrans_options,
    index=mtrans_options.index(st.session_state.get("MTRANS", default_values["MTRANS"]))
)

# ------------------------
# Update session state
# ------------------------
for k, v in feature_inputs.items():
    st.session_state[k] = v

# Fungsi preprocessing > disesuaikan dengan proses encoding
def correct_preprocessing(feature_dict):
    """Preprocessing yang sesuai dengan expected features model"""
    
    # Mapping yang lebih komprehensif
    categorical_to_numerical = {
        "Gender": {"Female": 0, "Male": 1},
        "FHWO": {"no": 0, "yes": 1},
        "FAVC": {"no": 0, "yes": 1},
        "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
        "SMOKE": {"no": 0, "yes": 1},
        "SCC": {"no": 0, "yes": 1},
        "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    }
    
    # Expected features - SESUAI DENGAN MODEL
    expected_features = [
        'ohe__MTRANS_Bike', 'ohe__MTRANS_Motorbike', 
        'ohe__MTRANS_Public_Transportation', 'ohe__MTRANS_Walking',
        'remainder__Gender', 'remainder__Age', 'remainder__Height', 
        'remainder__Weight', 'remainder__FHWO', 'remainder__FAVC', 
        'remainder__FCVC', 'remainder__NCP', 'remainder__CAEC', 
        'remainder__SMOKE', 'remainder__CH2O', 'remainder__SCC', 
        'remainder__FAF', 'remainder__TUE', 'remainder__CALC'
    ]
    
    # Inisialisasi dengan nilai default 0
    processed_data = {feature: 0.0 for feature in expected_features}
    
    # 1. Handle MTRANS one-hot encoding - KOREKSI PENTING!
    mtrans_mapping = {
        "Bike": {"ohe__MTRANS_Bike": 1, "ohe__MTRANS_Motorbike": 0, 
                "ohe__MTRANS_Public_Transportation": 0, "ohe__MTRANS_Walking": 0},
        "Motorbike": {"ohe__MTRANS_Bike": 0, "ohe__MTRANS_Motorbike": 1, 
                     "ohe__MTRANS_Public_Transportation": 0, "ohe__MTRANS_Walking": 0},
        "Public_Transportation": {"ohe__MTRANS_Bike": 0, "ohe__MTRANS_Motorbike": 0, 
                                "ohe__MTRANS_Public_Transportation": 1, "ohe__MTRANS_Walking": 0},
        "Walking": {"ohe__MTRANS_Bike": 0, "ohe__MTRANS_Motorbike": 0, 
                   "ohe__MTRANS_Public_Transportation": 0, "ohe__MTRANS_Walking": 1},
        "Automobile": {"ohe__MTRANS_Bike": 0, "ohe__MTRANS_Motorbike": 0, 
                      "ohe__MTRANS_Public_Transportation": 0, "ohe__MTRANS_Walking": 0}
    }
    
    # Apply MTRANS encoding
    mtrans_value = feature_dict.get("MTRANS", "Automobile")
    if mtrans_value in mtrans_mapping:
        for feature, value in mtrans_mapping[mtrans_value].items():
            processed_data[feature] = value
    else:
        # Default to Automobile jika tidak dikenal
        for feature, value in mtrans_mapping["Automobile"].items():
            processed_data[feature] = value
    
    # 2. Handle numerical features dengan validasi dan batasan
    numerical_features = {
        "Age": ("remainder__Age", 14, 61),
        "Height": ("remainder__Height", 1.45, 1.98),
        "Weight": ("remainder__Weight", 39, 173),
        "FCVC": ("remainder__FCVC", 1, 3),
        "NCP": ("remainder__NCP", 1, 4),
        "CH2O": ("remainder__CH2O", 1, 3),
        "FAF": ("remainder__FAF", 0, 3),
        "TUE": ("remainder__TUE", 0, 2)
    }
    
    for feature_name, (processed_name, min_val, max_val) in numerical_features.items():
        try:
            value = float(feature_dict.get(feature_name, min_val))
            # Clamp values to valid range
            value = max(min_val, min(value, max_val))
            processed_data[processed_name] = value
        except (ValueError, TypeError) as e:
            print(f"Error converting {feature_name}: {e}, using default: {min_val}")
            processed_data[processed_name] = min_val
    
    # 3. Handle categorical features dengan validasi
    categorical_features = {
        "Gender": "remainder__Gender",
        "FHWO": "remainder__FHWO", 
        "FAVC": "remainder__FAVC",
        "CAEC": "remainder__CAEC",
        "SMOKE": "remainder__SMOKE",
        "SCC": "remainder__SCC",
        "CALC": "remainder__CALC"
    }
    
    for feature_name, processed_name in categorical_features.items():
        value = feature_dict.get(feature_name, list(categorical_to_numerical[feature_name].keys())[0])
        processed_data[processed_name] = categorical_to_numerical[feature_name].get(value, 0)
    
    # 4. Create DataFrame dengan urutan yang tepat
    input_df = pd.DataFrame([processed_data])[expected_features]
       
    return input_df

# =====================================================================================================
# TAB LAYOUT
# =====================================================================================================
tab1, tab2, tab3 = st.tabs(["🎯 Prediction", "📊 Model Performance", "ℹ️ About"])

with tab1: # Main tab: Prediction of model
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ----------------------------------------
        # Summary information of input parameters
        # ----------------------------------------
        st.subheader("📋 Your Input Summary")
        
        # Personal Information
        st.write(
            """
            <div style='background-color: #F5F5F5; padding: 10px; border-radius: 5px; color: black;'>
                Personal Information
            </div>
            """,
            unsafe_allow_html=True
        )
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Gender", feature_inputs["Gender"])
        with col_b:
            st.metric("Age (y.o)", f"{feature_inputs['Age']}")
        with col_c:
            st.metric("Height (cm)", f"{feature_inputs["Height"]}")
        with col_d:
            st.metric("Weight (kg)", f"{feature_inputs["Weight"]}")
            
        # Health Indicators
        st.write(
            """
            <div style='background-color: #F5F5F5; padding: 10px; border-radius: 5px; color: black;'>
                Health Indicators
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Family History of Overweight**: {feature_inputs['FHWO']}")
            st.write(f"**Smoking Habit**: {feature_inputs['SMOKE']}")
        with col_b:
            st.write(f"**Alcohol Consumption**: {feature_inputs['CALC']}")
        
        # Dietary habit
        st.write(
            """
            <div style='background-color: #F5F5F5; padding: 10px; border-radius: 5px; color: black;'>
                Dietary Habits
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Vegetable Intake**: {feature_inputs['FCVC']}/3.0")
            st.write(f"**Meals per Day**: {feature_inputs['NCP']}/4.0")
            st.write(f"**High-Calorie Food**: {feature_inputs['FAVC']}")                      
        with col_b:           
            st.write(f"**Food between meals**: {feature_inputs['CAEC']}")
            st.write(f"**Calorie monitoring**: {feature_inputs['SCC']}")
        
        # Lifestyle Factors
        st.write(
            """
            <div style='background-color: #F5F5F5; padding: 10px; border-radius: 5px; color: black;'>
                Lifestyle Factors
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.write(f"**Physical Activity**: {feature_inputs['FAF']}/3.0")
            st.write(f"**Water Consumption**: {feature_inputs['CH2O']}/3.0")
        with col_b:
            st.write(f"**Screen Time**: {feature_inputs['TUE']}/2.0")
            st.write(f"**Transportation**: {feature_inputs['MTRANS']}")
    
    with col2:
    
        # -----------------------------
        # BMI Analysis
        # -----------------------------
        st.subheader("⚖️ BMI Analysis")
        
        # BMI description in short
        st.write(
            """
            <div style='background-color: #F5F5F5; padding: 10px; border-radius: 5px; color: black;'>
                Evaluate your weight category based on height and weight
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.metric("Body Mass Index (kg/m²)", f"{bmi:.1f}")
        if bmi < 18.5:
            st.info("📊 **Category**: Underweight")
            st.progress(0.3)
        elif bmi < 25:
            st.success("📊 **Category**: Normal weight")
            st.progress(0.5)
        elif bmi < 30:
            st.warning("📊 **Category**: Overweight")
            st.progress(0.7)
        else:
            st.error("📊 **Category**: Obesity")
            st.progress(0.9)
       
    # --------------------------------------------------------------------------------
    # Prediction button
    # --------------------------------------------------------------------------------
    if st.button("🎯 Predict Obesity Level", type="primary", use_container_width=True):
        with st.spinner("Analyzing features and making prediction..."):
            try:
                input_df = correct_preprocessing(feature_inputs)
                dmatrix = xgb.DMatrix(input_df)
                prediction = model.predict(dmatrix)
                
                if len(prediction.shape) > 1 and prediction.shape[1] > 1:
                    pred_proba = prediction[0]
                    pred_class = np.argmax(pred_proba)
                else:
                    pred_class = int(prediction[0])
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
                
                # --------------
                # VISUALISASI
                # --------------
                st.subheader("📊 Visualization Dashboard")
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Row 1: (1) Gauge Chart & (2) Obesity Levels
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_gauge_chart(pred_class, pred_proba, class_mapping), 
                                  use_container_width=True)
                
                with col2:
                    risk_fig, risk_info = create_risk_meter_with_legend(pred_class)
                    st.plotly_chart(risk_fig, use_container_width=True)
                    display_color_bar_legend(risk_info)

                st.markdown("---")
                
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Row 2: Donut Chart & bar chart
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                st.markdown("### 📊 Obesity Level Probabilities")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_donut_chart(pred_proba, class_mapping), 
                                  use_container_width=True)
                with col2:
                    # Langsung definisikan obesity_levels
                    # import plotly.graph_objects as go

                    obesity_levels = [
                        'Insufficient Weight',
                        'Normal Weight', 
                        'Overweight Level I',
                        'Overweight Level II',
                        'Obesity Type I',
                        'Obesity Type II', 
                        'Obesity Type III'
                    ]

                    if len(pred_proba) == 7:
                        colors = ['#4ECDC4', '#45B7D1', '#FFD166', '#FF9F1C', '#FF6B6B', '#EE4266', '#C44569']
                        
                        # BUAT BAR CHART DENGAN PLOTLY - Lebih responsive
                        fig = go.Figure()
                        
                        for i, (level, prob, color) in enumerate(zip(obesity_levels, pred_proba, colors)):
                            fig.add_trace(go.Bar(
                                x=[level],
                                y=[prob],
                                name=level,
                                marker_color=color,
                                opacity=0.8,
                                hovertemplate=f'<b>{level}</b><br>Probability: {prob:.3f}<extra></extra>'
                            ))
                        
                        # Highlight predicted class
                        if pred_class < len(pred_proba):
                            fig.data[pred_class].marker.line.color = 'black'
                            fig.data[pred_class].marker.line.width = 3
                            fig.data[pred_class].opacity = 1.0
                        
                        fig.update_layout(
                            #title="📊 Obesity Level Probabilities",
                            xaxis_title="Obesity Levels",
                            yaxis_title="Probability",
                            showlegend=False,
                            height=500,  # Tinggi optimal
                            margin=dict(t=50, l=50, r=50, b=100),  # Margin untuk label panjang
                            xaxis=dict(tickangle=45)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error(f"Expected 7 probability values, got {len(pred_proba)}")
                
                st.markdown("---")
                
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # Row 3: Tampilkan 3 radar chart dalam columns
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                st.subheader("📊 Health Profile Analysis")
                              
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.plotly_chart(create_health_radar(feature_inputs), use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_dietary_radar(feature_inputs), use_container_width=True)
                
                with col3:
                    st.plotly_chart(create_lifestyle_radar(feature_inputs), use_container_width=True)             
                
                st.markdown("---")
                
                # -----------------------
                # Health recommendations
                # -----------------------
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
                rec_list = recommendations.get(rec_key, [])

                # Build HTML sekaligus biar clean
                recommendations_html = "".join([f'<div style="margin-left: 40px; margin-bottom: 8px;">✅ {rec}</div>' for rec in rec_list])

                st.markdown(f'<div>{recommendations_html}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("💡 Tips: Make sure all input values are within the specified ranges.")

with tab2:
    st.header("📊 Model Performance Metrics")
    
    st.info("""
    **Model Performance from Training Evaluation:**
    - These metrics show how the model performed on test data during training
    - They represent the model's overall reliability
    - For your specific prediction, check the confidence score in Prediction tab
    """)
    # -----------------------
    # Classification Report
    # -----------------------
    st.subheader("📈 Classification Report")
    header_image_url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/assets/classification-report.png"
    st.image(header_image_url, width=800)
    st.markdown("---")
    
    # ---------------------------------
    # Model performance comparison
    # ---------------------------------
    st.subheader("💹 Model Performance Comparison")
    header_image_url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/assets/model-performance-comparison.png"
    st.image(header_image_url, width=1000)
    st.markdown("---")
    
    # ---------------------------------
    # Confusion Matrix: Upload png file
    # ---------------------------------
    st.subheader("🎯 Confusion Matrix from Model Training")
    st.write("""
    **How to read this matrix:**
    - **Diagonal (blue squares)**: Correct predictions  
    - **Off-diagonal**: Misclassifications
    - **Rows**: True obesity classes
    - **Columns**: Predicted obesity classes
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        header_image_url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/assets/confusion_matrix.png"
        st.image(header_image_url, width=700)
    with col2:
        st.write("""      
        **Obesity Level:**
        - 0 : Insufficient Weight
        - 1 : Normal Weight
        - 2 : Overweight Level I 
        - 3 : Overweight Level II
        - 4 : Obesity Type I
        - 5 : Obesity Type II
        - 6 : Obesity Type III
        """)
    st.markdown("---")
    
    # ---------------------------------
    # Top 10 Important Features
    # ---------------------------------
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 10 Important Features")
        
        header_image_url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/assets/importances.png"
        st.image(header_image_url, width=800)  
    with col2:
        st.subheader("Columm Description")
        
        header_image_url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/assets/columns-dataset.png"
        st.image(header_image_url, width=800)  
    st.markdown("---")
    
    # ---------------------------------
    # Model Analysis
    # ---------------------------------
    st.subheader("🔍 Model Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Model Strengths:**
        - High overall accuracy: All metrics (Precision, Recall, F1-Score) exceed 90%, indicating excellent predictive performance.
        - Consistent performance across all obesity level classes with no significant drop in accuracy.
        - Balanced precision and recall, showing that misclassifications are minimal.
        - Strong generalization ability, suggesting the model is not overfitted to any specific class.
        - Deployment-ready model with stable, reliable, and well-balanced performance across all categories.
        """)
    
    with col2:
        st.markdown("""
        **📈 Areas for Improvement:**
        - Possible overfitting — model may perform too perfectly on test data.
        - Need more diverse and larger dataset for better generalization.
        - Slight misclassification between mid-level obesity classes (Normal, Overweight I).
        - Further feature analysis could improve class separation.
        """)

with tab3:
    st.header("ℹ️ About This Application")
    
    st.markdown("""
    ### 🏥 Obesity levels Prediction App 
    This application predicts obesity levels based on physical condition, health indicators, dietary habits, and lifestyle factors
    using a trained XGBoost machine learning model.
    
    #### 📋 Features Used:
    - **Personal Information**: Gender, Age, Height, Weight
    - **Health Indicators**: Family history with overweight, Smoking habit, alcohol consumption
    - **Dietary Habits**: Vegetable consumption, Meal frequency, High-calorie food intake, Food between meals, Calorie monitoring
    - **Lifestyle Factors**: Physical activity, Water intake, Technology usage, Transportation
  
    #### 🎯 Obesity Levels:
    - Insufficient Weight
    - Normal Weight  
    - Overweight Level I & II
    - Obesity Type I, II & III
    
    #### 🔧 Technical Details:
    - **Model**: XGBoost Classifier
    - **Training Data**: Obesity dataset with multiple lifestyle factors
    - **Accuracy**: High predictive performance for obesity classification
    
    #### 📊 How to Read the Charts:
    - **Gauge Chart**: Shows model confidence in the prediction (0-100%)
    - **Donut Chart**: Probability distribution across all obesity levels
    - **Radar Chart**: Analysis of lifestyle factors and habits
    - **Feature Importance**: Shows which factors most influence predictions
    - **Confusion Matrix**: Model performance across different obesity classes
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
    "🏥 Obesity Levels Prediction App (OLPA) | Made with Streamlit & XGBoost"
    "</div>"
    "<div style='text-align: center; color: salmon; font-size: 14px; margin-top: 5px;'>"
    "© daudrusyadnurdin@gmail.com, DA/DS bootcamp 34, dibimbing.id"
    "</div>",
    unsafe_allow_html=True
)
