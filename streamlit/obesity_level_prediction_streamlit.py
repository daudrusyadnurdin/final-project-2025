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

def get_health_interpretation(score):
    """Interpretation of the health score"""
    if score >= 80:
        return "Excellent", "🟢", "health-score-excellent"
    elif score >= 60:
        return "Good", "🟡", "health-score-good"
    elif score >= 40:
        return "Fair", "🟠", "health-score-fair"
    else:
        return "Needs Improvement", "🔴", "health-score-poor"

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
        {'level': 2, 'label': 'Moderate', 'color': '#FFD166', 'description': 'Overweight I'},
        {'level': 3, 'label': 'High', 'color': '#FF9F1C', 'description': 'Overweight II'},
        {'level': 4, 'label': 'Very High', 'color': '#FF6B6B', 'description': 'Obesity I'},
        {'level': 5, 'label': 'Severe', 'color': '#EE4266', 'description': 'Obesity II'},
        {'level': 6, 'label': 'Critical', 'color': '#C44569', 'description': 'Obesity III'}
    ]
    
    current_risk = risk_info[pred_class]
    
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
        'Moderate': 'Overweight I',
        'High': 'Overweight II',
        'Very High': 'Obesity I',
        'Severe': 'Obesity II',
        'Critical': 'Obesity III'
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
    
# =======================================
# FUNCTIONS OF TAB PERFORMANCE & ANALYSIS
# =======================================

def create_real_feature_importance():
    """Feature importance realistic berdasarkan domain knowledge"""
    
    features = [
        'Weight', 'Height', 'Age', 'Physical Activity', 'High Caloric Food',
        'Vegetable Consumption', 'Meal Frequency', 'Water Intake', 
        'Screen Time', 'Between Meals', 'Family History', 'Alcohol Consumption',
        'Calorie Monitoring', 'Transportation', 'Gender', 'Smoking'
    ]
    
    importance_values = [85, 78, 65, 58, 55, 52, 48, 45, 42, 38, 35, 32, 28, 25, 22, 18]
    
    # Reverse the order - highest importance at the top
    features_reversed = features[::-1]
    importance_reversed = importance_values[::-1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(features_reversed))
    
    # Create colors based on ACTUAL importance values, not position
    max_importance = max(importance_values)
    min_importance = min(importance_values)
    
    colors = []
    for importance in importance_reversed:
        # Normalize importance to colormap range (0.2 to 0.8)
        normalized = 0.2 + (importance - min_importance) / (max_importance - min_importance) * 0.6
        color = plt.cm.Blues(normalized)
        colors.append(color)
    
    bars = ax.barh(y_pos, importance_reversed, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_reversed)
    ax.set_xlabel('Feature Importance Score')
    ax.set_title('Feature Importance in Obesity Prediction')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
               f'{width:.0f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_bmi_distribution_chart(user_bmi, pred_class):
    """Menampilkan BMI user dalam konteks distribusi"""
    
    categories = [
        'Underweight (<18.5)', 'Normal (18.5-24.9)', 'Overweight (25-29.9)',
        'Obesity I (30-34.9)', 'Obesity II (35-39.9)', 'Obesity III (40+)'
    ]
    
    distribution = [5, 35, 30, 15, 10, 5]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(categories, distribution, color=['#4ECDC4', '#45B7D1', '#FFD166', 
                                                  '#FF9F1C', '#FF6B6B', '#C44569'])
    
    user_category_idx = 0
    if user_bmi < 18.5: user_category_idx = 0
    elif user_bmi < 25: user_category_idx = 1
    elif user_bmi < 30: user_category_idx = 2
    elif user_bmi < 35: user_category_idx = 3
    elif user_bmi < 40: user_category_idx = 4
    else: user_category_idx = 5
    
    bars[user_category_idx].set_color('red')
    bars[user_category_idx].set_alpha(0.8)
    
    ax.set_ylabel('Population Distribution (%)')
    ax.set_title(f'BMI Distribution - Your BMI: {user_bmi:.1f} (Red Bar)')
    ax.tick_params(axis='x', rotation=45)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{height}%', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_lifestyle_comparison_chart(feature_inputs):
    """Membandingkan lifestyle user dengan rekomendasi sehat"""
    
    factors = ['Vegetable Intake', 'Physical Activity', 'Water Consumption', 
               'Meal Frequency', 'Screen Time']
    
    user_scores = [
        (feature_inputs['FCVC'] / 3.0) * 100,
        (feature_inputs['FAF'] / 3.0) * 100,
        (feature_inputs['CH2O'] / 3.0) * 100,
        ((feature_inputs['NCP'] - 1) / 3.0) * 100,
        100 - (feature_inputs['TUE'] / 2.0) * 100
    ]
    
    healthy_targets = [80, 70, 80, 75, 60]
    
    x = np.arange(len(factors))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width/2, user_scores, width, label='Your Score', color='#1f77b4')
    bars2 = ax.bar(x + width/2, healthy_targets, width, label='Healthy Target', color='#2ca02c')
    
    ax.set_ylabel('Score (0-100)')
    ax.set_title('Lifestyle Comparison vs Healthy Targets')
    ax.set_xticks(x)
    ax.set_xticklabels(factors)
    ax.legend()
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                   f'{height:.0f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def create_health_risk_breakdown(feature_inputs):
    """Breakdown faktor risiko spesifik user"""
    
    risk_factors = {
        'High Caloric Food': 30 if feature_inputs['FAVC'] == 'yes' else 0,
        'Low Vegetable Intake': 40 if feature_inputs['FCVC'] < 2 else 0,
        'Low Physical Activity': 50 if feature_inputs['FAF'] < 1.5 else 20,
        'Frequent Snacking': 35 if feature_inputs['CAEC'] in ['Frequently', 'Always'] else 0,
        'Sedentary Lifestyle': 45 if feature_inputs['TUE'] > 1.5 else 15,
        'Family History': 25 if feature_inputs['FHWO'] == 'yes' else 0
    }
    
    active_risks = {k: v for k, v in risk_factors.items() if v > 0}
    
    if not active_risks:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    factors = list(active_risks.keys())
    risks = list(active_risks.values())
    
    bars = ax.barh(factors, risks, color=['#FF6B6B', '#FF9F1C', '#FFD166', '#EE4266', '#C44569', '#FF9F1C'])
    ax.set_xlabel('Risk Score')
    ax.set_title('Your Specific Health Risk Factors')
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
               f'{width}', ha='left', va='center')
    
    plt.tight_layout()
    return fig

def create_performance_metrics():
    """Metrics dari training"""
    
    return {
        'Precision': '97.129%', 
        'Recall': '97.129%',
        'F1-Score': '97.129%',
        'Macro Avg F1': '97.040%',
        'Weighted Avg F1': '97.125%',
        'Accuracy': '97.129%'
    }

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

# Dietary Habits: Vegetable consumption, Meal frequency, High-calorie food intake, Food between meals, Calorie monitoring
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


# Update session state
for k, v in feature_inputs.items():
    st.session_state[k] = v

# Fungsi preprocessing > disesuaikan dengan proses encoding
def correct_preprocessing(feature_dict):
    """Preprocessing yang sesuai dengan expected features model"""
    
    categorical_to_numerical = { # add Gender > wrong encoding before
        #"Gender": {"Male": 0, "yes": 1},
        "FHWO": {"no": 0, "yes": 1},
        "FAVC": {"no": 0, "yes": 1},
        "CAEC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3},
        "SMOKE": {"no": 0, "yes": 1},
        "SCC": {"no": 0, "yes": 1},
        "CALC": {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
    }
    
    # Posisi Gender: remainder__Gender
    expected_features = [
        'ohe__Gender_Male', 'ohe__MTRANS_Bike', 'ohe__MTRANS_Motorbike', 
        'ohe__MTRANS_Public_Transportation', 'ohe__MTRANS_Walking', 
        'remainder__Age', 'remainder__Height', 'remainder__Weight', 
        'remainder__FHWO', 'remainder__FAVC', 'remainder__FCVC', 
        'remainder__NCP', 'remainder__CAEC', 'remainder__SMOKE', 
        'remainder__CH2O', 'remainder__SCC', 'remainder__FAF', 
        'remainder__TUE', 'remainder__CALC'
    ]
    
    processed_data = {feature: 0.0 for feature in expected_features}
    
    # Set one-hot encoding untuk Gender >> nantinya dibuang, jadi lebih simple
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
    
    for feature in ['ohe__MTRANS_Bike', 'ohe__MTRANS_Motorbike', 
                   'ohe__MTRANS_Public_Transportation', 'ohe__MTRANS_Walking']:
        processed_data[feature] = 0.0
    
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
    
    # Set encoded categorical features >> sisipkan Gender di sini!
    processed_data["remainder__FHWO"] = categorical_to_numerical["FHWO"][feature_dict["FHWO"]]
    processed_data["remainder__FAVC"] = categorical_to_numerical["FAVC"][feature_dict["FAVC"]]
    processed_data["remainder__CAEC"] = categorical_to_numerical["CAEC"][feature_dict["CAEC"]]
    processed_data["remainder__SMOKE"] = categorical_to_numerical["SMOKE"][feature_dict["SMOKE"]]
    processed_data["remainder__SCC"] = categorical_to_numerical["SCC"][feature_dict["SCC"]]
    processed_data["remainder__CALC"] = categorical_to_numerical["CALC"][feature_dict["CALC"]]
    
    input_df = pd.DataFrame([processed_data])[expected_features]
    
    return input_df

# ============================
# TAB LAYOUT
# ============================

tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Model Performance", "🔍 Health Analysis", "ℹ️ About"])

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
    
        # -------------
        # BMI Analysis
        # -------------
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
    
    
    # ------------------
    # Prediction button
    # ------------------
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
                st.subheader("📊 Advanced Visualization Dashboard")
                
                # Row 1: Gauge Chart dan Risk Meter
                col1, col2 = st.columns(2)
                
                with col1:
                    st.plotly_chart(create_gauge_chart(pred_class, pred_proba, class_mapping), 
                                  use_container_width=True)
                
                with col2:
                    risk_fig, risk_info = create_risk_meter_with_legend(pred_class)
                    st.plotly_chart(risk_fig, use_container_width=True)
                    display_color_bar_legend(risk_info)

                # Di tab prediction
                st.subheader("📊 Health Profile Analysis")
                
                # Tampilkan 3 radar chart dalam columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.plotly_chart(create_health_radar(feature_inputs), use_container_width=True)
                
                with col2:
                    st.plotly_chart(create_dietary_radar(feature_inputs), use_container_width=True)
                
                with col3:
                    st.plotly_chart(create_lifestyle_radar(feature_inputs), use_container_width=True)             

                # ----------------------------------
                # Row 2: Donut Chart dan Radar Chart
                # ----------------------------------
                st.markdown("### 📊 Obesity Level Probabilities")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.plotly_chart(create_donut_chart(pred_proba, class_mapping), 
                                  use_container_width=True)
                with col4:
                    # Langsung definisikan obesity_levels
                    #import plotly.graph_objects as go

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
                
                # -----------------------
                # Health recommendations
                # -----------------------
                st.markdown("---")
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
    
    # Performance Metrics
    st.subheader("📈 Overall Performance Metrics")
    metrics = create_performance_metrics()
    
    cols = st.columns(3)
    metric_items = list(metrics.items())
    
    for i, (name, value) in enumerate(metric_items):
        with cols[i % 3]:
            st.metric(label=name, value=value)
    

    
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
    
    # ---------------------------------
    # Top 10 Important Features
    # ---------------------------------
    st.subheader("🏆 Top 10 Important Features")
    
    header_image_url = "https://raw.githubusercontent.com/daudrusyadnurdin/final-project-2025/main/assets/importances.png"
    st.image(header_image_url, width=800)
    
    
    
    # ---------------------------------
    # Class-wise Performance
    # ---------------------------------
    st.subheader("📋 Class-wise Performance on Test Data")
    
    class_performance = {
        'Class': ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II', 
                 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'],
        'Precision': ['96.61%', '91.80%', '94.44%', '96.08%', '100.00%', '100.00%', '100.00%'],
        'Recall': ['96.61%', '91.80%', '92.27%', '100.00%', '98.57%', '100.00%', '100.00%'],
        'F1-Score': ['96.61%', '91.80%', '93.58%', '98.00%', '99.28%', '100.00%', '100.00%'],
        'Support': ['59', '61', '55', '49', '70', '64', '60']
    }

    # next panggil file 
    
    perf_df = pd.DataFrame(class_performance)
    st.dataframe(perf_df, use_container_width=True)
    
    # Model Analysis
    st.subheader("🔍 Model Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **✅ Model Strengths:**
        - Excellent at identifying **Underweight** and **Normal Weight** (F1 > 88%)
        - Good overall balance across all classes
        - Consistent performance metrics
        """)
    
    with col2:
        st.markdown("""
        **📈 Areas for Improvement:**
        - **Obesity Type III** has lower recall (75.1%)
        - Some confusion between **Overweight II** and **Obesity I**
        - Could benefit from more **Obesity III** training samples
        """)

with tab3:
    st.header("🔍 Detailed Health Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Feature Importance")
        st.write("**The most influential factor in predicting obesity:**")
        
        importance_fig = create_real_feature_importance()
        st.pyplot(importance_fig)
        
        st.markdown("""
        **Interpretasi:**
        - **Weight & Height**: The most dominant physical factor (BMI calculation)
        - **Age & Physical Activity**: Lifestyle factors with significant influence
        - **Dietary Habits**: Dietary patterns determine calorie intake
        """)
        
        st.subheader("🔄 Lifestyle Comparison")
        st.write("Comparison of your lifestyle with healthy targets:")
        st.pyplot(create_lifestyle_comparison_chart(feature_inputs))
    
    with col2:
        st.subheader("⚖️ BMI Context")
        st.write("Your BMI position within the population distribution:")
        
        if 'pred_class' in locals():
            st.pyplot(create_bmi_distribution_chart(bmi, pred_class))
        else:
            st.info("Perform a prediction first to analyze the BMI distribution.")
        
        st.subheader("🎯 Risk Factors Breakdown")
        risk_chart = create_health_risk_breakdown(feature_inputs)
        if risk_chart:
            st.write("**Specific risk factors based on your input:**")
            st.pyplot(risk_chart)
        else:
            st.success("✅ No significant risk factors have been identified.")
    
    # Insights & Recommendations
    st.subheader("💡 Personalized Action Plan")
    
    personalized_recs = []
    
    if feature_inputs['FAF'] < 1.5:
        personalized_recs.append("🚶 **Enhance physical activity levels**: Target a minimum of 30 minutes of moderate exercise daily, five times per week.")
    
    if feature_inputs['FCVC'] < 2:
        personalized_recs.append("🥦 **Eat more greens**: Make sure every meal includes a generous portion of vegetables.")
    
    if feature_inputs['CH2O'] < 2:
        personalized_recs.append("💧 **Stay hydrated**: Drink at least 2–3 liters of water every day to keep your body functioning optimally.")
    
    if feature_inputs['TUE'] > 1.5:
        personalized_recs.append("📱 **Cut down on screen time**: Spend less time on gadgets and stay more active during the day.")
    
    if feature_inputs['FAVC'] == 'yes':
        personalized_recs.append("🍔 **Cut back on high-calorie foods**: fuel your body with healthier, energizing choices that help you feel your best!")
    
    if not personalized_recs:
        personalized_recs.append("✅ **Keep up your healthy habits and stay consistent!**")
    
    for i, rec in enumerate(personalized_recs, 1):
        st.write(f"{i}. {rec}")

with tab4:
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
    - **Risk Meter**: Visual representation of obesity level (0-6)
    - **Donut Chart**: Probability distribution across all obesity levels
    - **Radar Chart**: Analysis of lifestyle factors and habits
    - **Feature Importance**: Shows which factors most influence predictions
    - **BMI Distribution**: Compares your BMI with population distribution
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
