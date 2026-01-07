import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt

# ------------------------------------
# PAGE CONFIGURATION
# ------------------------------------
st.set_page_config(
    page_title="Alzheimer’s Prediction",
    page_icon="🔬",
    layout="wide"
)

# ------------------------------------
# CUSTOM CSS FOR APP
# ------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');

.stApp { background-color: #bfc0c0 !important; }

/* Titles */
.main-title { font-size: 36px; font-family: 'Roboto', sans-serif; font-weight: 700; color: #0a4b78; text-align: center; margin-bottom: 5px; }
.subtitle { font-size: 18px; font-family: 'Roboto', sans-serif; font-weight: 400; color: #1b5e8f; text-align: center; margin-bottom: 30px; }
.intro-text {
    font-size: 16.5px;
    line-height: 1.55;
    background: rgba(255, 255, 255, 0.08);
    padding: 18px 20px;
    border-radius: 12px;
    margin-top: 12px;
    color: #1f1b1b;
    border-left: 4px solid #9b59b6;
}          
.section-heading { font-size: 28px; font-family: 'Roboto', sans-serif; font-weight: 600; color: #1a1a1a; text-align: center; margin-bottom: 15px; }
.card-title { font-size: 20px; font-family: 'Roboto', sans-serif; font-weight: 600; color: #1a1a1a; margin-bottom: 10px; }

/* Apply custom-label style to all Streamlit input labels */
div.stSlider > label,
div.stSelectbox > label,
div.stNumberInput > label,
div[data-testid="stSliderLabel"],
div[data-testid="stSelectboxLabel"],
label {
    color: #1a1a1a !important;
    font-weight: 400 !important;
    font-size: 14px !important;
}

/* Overview Cards */
.overview-card {
    padding: 20px; border-radius: 15px; color: white; font-family: 'Roboto', sans-serif;
    transition: transform 0.2s, box-shadow 0.2s; box-shadow: 2px 4px 10px rgba(0,0,0,0.15); margin-bottom: 15px;
}
.overview-card:hover { transform: translateY(-5px); box-shadow: 4px 8px 20px rgba(0,0,0,0.3); }
.blue { background-color: #577399; } .green { background-color: #9db4c0; }
.yellow { background-color: #9db4c0; color: black; } .pink { background-color: #577399; }
ul { margin: 0; padding-left: 20px; }

/* Input Cards */
.input-card {
    background-color: #bee3db; padding: 20px; border-radius: 15px;
    box-shadow: 2px 4px 15px rgba(0,0,0,0.1); margin-bottom: 20px;
    transition: transform 0.2s, box-shadow 0.2s;
}
.input-card:hover { transform: translateY(-3px); box-shadow: 4px 8px 25px rgba(0,0,0,0.2); }
.custom-label { font-weight: 500; font-size: 16px; margin-bottom: 5px; color: #1a1a1a; display:block; }

/* Custom Checkbox */
.ui-checkbox {
  --primary-color: #1677ff; --secondary-color: #fff; --primary-hover-color: #4096ff;
  --checkbox-diameter: 20px; --checkbox-border-radius: 5px;
  --checkbox-border-color: #d9d9d9; --checkbox-border-width: 1px; --checkbox-border-style: solid;
  --checkmark-size: 1.2;
}
.ui-checkbox, .ui-checkbox *, .ui-checkbox *::before, .ui-checkbox *::after { box-sizing: border-box; }
.ui-checkbox { appearance: none; width: var(--checkbox-diameter); height: var(--checkbox-diameter); border-radius: var(--checkbox-border-radius);
  background: var(--secondary-color); border: var(--checkbox-border-width) var(--checkbox-border-style) var(--checkbox-border-color);
  transition: all 0.3s; cursor: pointer; position: relative; margin-right: 5px;}
.ui-checkbox::after { content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
  box-shadow: 0 0 0 calc(var(--checkbox-diameter)/2.5) var(--primary-color); border-radius: inherit; opacity: 0; transition: all 0.5s cubic-bezier(0.12,0.4,0.29,1.46);}
.ui-checkbox::before { top: 40%; left: 50%; content: ""; position: absolute; width: 4px; height: 7px; border-right: 2px solid var(--secondary-color);
  border-bottom: 2px solid var(--secondary-color); transform: translate(-50%,-50%) rotate(45deg) scale(0); opacity: 0; transition: all 0.1s cubic-bezier(0.71,-0.46,0.88,0.6), opacity 0.1s;}
.ui-checkbox:hover { border-color: var(--primary-color); }
.ui-checkbox:checked { background: var(--primary-color); border-color: transparent;}
.ui-checkbox:checked::before { opacity: 1; transform: translate(-50%,-50%) rotate(45deg) scale(var(--checkmark-size)); transition: all 0.2s cubic-bezier(0.12,0.4,0.29,1.46) 0.1s; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------
# TITLE SECTION
# ------------------------------------
st.markdown("<div class='main-title'>🔬 Alzheimer’s Disease Risk Prediction</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Clinical-grade machine learning assistance for early screening</div>", unsafe_allow_html=True)
st.markdown("""
<div class='intro-text'>
Alzheimer’s disease is a progressive neurological disorder that affects memory, thinking, and behavior.
It is the most common cause of dementia, gradually impacting a person’s ability to perform daily activities.  
Early detection plays a crucial role in slowing progression, planning treatment, and improving quality of life.
</div>
""", unsafe_allow_html=True)
# ------------------------------------
# LOAD MODEL + SCALER + FEATURES
# ------------------------------------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
with open("selected_features.json", "r") as f:
    selected_features = json.load(f)

# ------------------------------------
# OVERVIEW CARDS
# ------------------------------------
st.markdown("<div class='section-heading'> Overview</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="overview-card blue">
        <div class="card-title">🔵 What Happens in Alzheimer’s?</div>
        <p>Brain cells (neurons) begin to deteriorate and lose communication.Abnormal protein buildup (amyloid & tau) disrupts brain functions.
           Memory loss, confusion, and behavioral changes appear gradually.The disease progresses through mild, moderate, and severe stages. </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="overview-card green">
        <div class="card-title">🟢 Early Signs to Watch For</div>
        <ul style="font-size:15px; color:#333; line-height:1.5;">
<li>Frequent memory lapses (forgetting names, dates, recent events)</li>
<li>Difficulty planning, solving problems, or focusing</li>
<li>Misplacing items and losing track of time or location</li>
<li>Reduced judgement or decision-making ability</li>
<li>Social withdrawal and changes in mood or personality</li>
</ul>
    </div>
    """, unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    st.markdown("""
    <div class="overview-card yellow">
        <div class="card-title">🟡 How the Model Works</div>
        <p>The XGBoost-based AI model analyzes medical, cognitive, lifestyle, and behavioral factors to estimate Alzheimer's risk with clinical-grade precision.<p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="overview-card pink">
        <div class="card-title">🩷 Clinical Advisory</div>
        <p>This tool supports—but does <i>not</i> replace—clinical diagnosis. Consult a neurologist for medical evaluation.</p>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# ------------------------------------
# USER INPUT FORM
# ------------------------------------
st.markdown("<div class='section-heading'> Enter Patient Assessment Data</div>", unsafe_allow_html=True)
colA, colB, colC = st.columns(3)
input_data = {}

# ---- Column A ----
with colA:
    st.markdown('<div class="input-card"><div class="card-title">❇ Patient Information 👤</div></div>', unsafe_allow_html=True)
    input_data["Age"] = st.slider("Age", 40, 100, 60)
    gender = st.selectbox("Gender", ["Male", "Female"])
    input_data["Gender"] = 1 if gender=="Male" else 0
    ethnicity_map = {"Caucasian":0,"African American":1,"Asian":2,"Other":3}
    eth_sel = st.selectbox("Ethnicity", list(ethnicity_map.keys()))
    input_data["Ethnicity"] = ethnicity_map[eth_sel]
    input_data["BMI [Body Mass Index]"] = st.slider("BMI", 10.0,40.0,22.5)

    # Smoking Yes/No
    st.markdown('<label class="custom-label">Smoking</label>', unsafe_allow_html=True)
    col_yes, col_no = st.columns(2)
    with col_yes:
        smoking_yes = st.checkbox("Yes", key="smoking_yes")
    with col_no:
        smoking_no = st.checkbox("No", key="smoking_no")
    if smoking_yes and smoking_no:
        st.warning("Please select either Yes or No for Smoking, not both.")
        input_data["Smoking"] = None
    elif smoking_yes:
        input_data["Smoking"] = 1
    elif smoking_no:
        input_data["Smoking"] = 0
    else:
        input_data["Smoking"] = None

    input_data["AlcoholConsumption"] = st.slider("Alcohol Consumption (units/week)",0.0,20.0,1.0)

# ---- Column B ----
with colB:
    st.markdown('<div class="input-card"><div class="card-title">❇ Cognitive & Functional</div></div>', unsafe_allow_html=True)
    input_data["MMSE [Mini-Mental state Examination Score]"] = st.slider("MMSE Score",0,30,20)
    input_data["ADL [Activities of Daily Living Score]"] = st.slider("ADL Score",0,60,30)
    input_data["FunctionalAssessment"] = st.slider("Functional Assessment",0,60,20)

    st.markdown('<div class="input-card"><div class="card-title">❇ Behavioral & Symptoms</div></div>', unsafe_allow_html=True)
    behavior_features = ["MemoryComplaints","Confusion","Disorientation","DifficultyCompletingTasks","Forgetfulness"]
    for feat in behavior_features:
        st.markdown(f'<label class="custom-label">{feat.replace("_"," ")}</label>', unsafe_allow_html=True)
        col_yes, col_no = st.columns(2)
        with col_yes:
            yes = st.checkbox("Yes", key=f"{feat}_yes")
        with col_no:
            no = st.checkbox("No", key=f"{feat}_no")
        if yes and no:
            st.warning(f"Select either Yes or No for {feat}, not both.")
            input_data[feat] = None
        elif yes:
            input_data[feat] = 1
        elif no:
            input_data[feat] = 0
        else:
            input_data[feat] = None

# ---- Column C ----
with colC:
    st.markdown('<div class="input-card"><div class="card-title">❇ Additional Indicators</div></div>', unsafe_allow_html=True)
    extra_features = ["BehavioralProblems","PersonalityChanges"]
    for feat in extra_features:
        st.markdown(f'<label class="custom-label">{feat.replace("_"," ")}</label>', unsafe_allow_html=True)
        col_yes, col_no = st.columns(2)
        with col_yes:
            yes = st.checkbox("Yes", key=f"{feat}_yes")
        with col_no:
            no = st.checkbox("No", key=f"{feat}_no")
        if yes and no:
            st.warning(f"Select either Yes or No for {feat}, not both.")
            input_data[feat] = None
        elif yes:
            input_data[feat] = 1
        elif no:
            input_data[feat] = 0
        else:
            input_data[feat] = None

    input_data["SleepQuality"] = st.slider("Sleep Quality (0=Poor, 5=Good)",0,5,2)

    st.markdown('<div class="input-card"><div class="card-title">❇ Medical History 🩺</div></div>', unsafe_allow_html=True)
    medical_features = ["FamilyHistoryAlzheimers","Hypertension","CardiovascularDisease","Diabetes","Depression","HeadInjury"]
    for feat in medical_features:
        st.markdown(f'<label class="custom-label">{feat.replace("_"," ")}</label>', unsafe_allow_html=True)
        col_yes, col_no = st.columns(2)
        with col_yes:
            yes = st.checkbox("Yes", key=f"{feat}_yes")
        with col_no:
            no = st.checkbox("No", key=f"{feat}_no")
        if yes and no:
            st.warning(f"Select either Yes or No for {feat}, not both.")
            input_data[feat] = None
        elif yes:
            input_data[feat] = 1
        elif no:
            input_data[feat] = 0
        else:
            input_data[feat] = None

st.write("---")

# ------------------------------------
# PREDICTION FUNCTION
# ------------------------------------
def run_prediction(data):
    df_input = pd.DataFrame([data])
    df_input = df_input[selected_features]
    scaled = scaler.transform(df_input)
    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][1]
    return pred, proba, scaled

# ------------------------------------
# PREDICT BUTTON
# ------------------------------------
st.markdown("""
<style>
/* 🔵 Change normal button color */
div.stButton > button {
    background-color: #1e88e5 !important;   /* Blue */
    color: white !important;
    border-radius: 10px !important;
    padding: 10px 18px !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    border: none !important;
}

/* 🟣 Change hover color */
div.stButton > button:hover {
    background-color: #0d47a1 !important;   /* Dark blue */
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# =============================
#       PREDICT BUTTON
# =============================
if st.button(" Predict Alzheimer’s Risk", use_container_width=True):

    pred, proba, scaled_input = run_prediction(input_data)

    # =============================
    #       PREDICTION RESULT
    # =============================
    st.markdown("<div class='section-heading'> Prediction Result</div>", unsafe_allow_html=True)

    # Stylish card for prediction result
    if pred == 1:
        st.markdown(f"""
        <div style="background-color:#ffebee; padding:20px; border-left:8px solid #c62828; border-radius:10px;">
            <h3 style="color:#b71c1c; margin-bottom:10px;">🔴 High Likelihood of Alzheimer’s</h3>
            <p style="color:#b71c1c; font-size:18px;">
                <b>Predicted Probability:</b> {proba:.2f}<br>
                Your result indicates a <b>significant risk level</b>.  
            </p>
            <p style="font-size:15px; color:#7f0000;">
                The model has detected patterns associated with early Alzheimer's indicators.
                Further medical evaluation is strongly recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="background-color:#e8f5e9; padding:20px; border-left:8px solid #2e7d32; border-radius:10px;">
            <h3 style="color:#1b5e20; margin-bottom:10px;">🟢 Low Likelihood of Alzheimer’s</h3>
            <p style="color:#1b5e20; font-size:18px;">
                <b>Predicted Probability:</b> {proba:.2f}<br>
                Your result suggests a <b>low probability</b> of Alzheimer’s.
            </p>
            <p style="font-size:15px; color:#0d3c08;">
                No immediate cognitive concerns detected.  
                Continue maintaining a brain-healthy lifestyle.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # =============================
    #     CLINICAL INTERPRETATION
    # =============================
    st.markdown("<div class='section-heading'> Clinical Interpretation</div>", unsafe_allow_html=True)

    if proba > 0.85:
        st.markdown("""
        <div style='background-color:#ffebee; padding:15px; border-left:6px solid #b71c1c; border-radius:8px;'>
            <h4 style='color:#b71c1c;'>🔻 Very High Risk (≥ 85%)</h4>
            <p style='color:#7f0000; font-size:15px;'>
            Strong indication of potential Alzheimer’s-related decline.  
            Immediate neurological consultation is advised.  
            Early detection may support treatment planning.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif proba > 0.65:
        st.markdown("""
        <div style='background-color:#fff3e0; padding:15px; border-left:6px solid #ef6c00; border-radius:8px;'>
            <h4 style='color:#e65100;'>🟧 High Risk (65% – 85%)</h4>
            <p style='color:#6d3000; font-size:15px;'>
            Higher-than-normal likelihood of cognitive decline.  
            A cognitive screening test and medical follow-up are recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)

    elif proba > 0.45:
        st.markdown("""
        <div style='background-color:#f1f8e9; padding:15px; border-left:6px solid #7cb342; border-radius:8px;'>
            <h4 style='color:#558b2f;'>🟡 Mild Risk (45% – 65%)</h4>
            <p style='color:#3a5d22; font-size:15px;'>
            Some mild indicators are present.  
            Lifestyle improvements and periodic checkups are recommended.
            </p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style='background-color:#e3f2fd; padding:15px; border-left:6px solid #1565c0; border-radius:8px;'>
            <h4 style='color:#0d47a1;'>🟦 Low Risk (≤ 45%)</h4>
            <p style='color:#0d3d77; font-size:15px;'>
            No significant cognitive risk patterns detected.  
            Maintain healthy physical, social, and cognitive habits.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")

# =============================
#            FOOTER
# =============================
st.markdown("""
<style>
.footer {
    font-size: 14px;
    color: #555555;
    text-align: center;
    margin-top: 40px;
    padding: 15px 10px;
    border-top: 2px solid #ddd;
    background-color: #f9f9f9;
    border-radius: 10px;
}
.footer b { color: #b71c1c; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    ⚠️ <b>Disclaimer:</b> This tool is a machine learning model designed to provide risk estimation for educational and informational purposes only.  
    It <b>does not replace professional medical advice, diagnosis, or treatment</b>.  
    Always consult a qualified healthcare provider for any concerns regarding Alzheimer's disease or other cognitive conditions.
</div>
""", unsafe_allow_html=True)




